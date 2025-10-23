from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import copy
import csv
import numpy as np
import random
from Queryset import Queryset, QueryDataset
from QuerySampler import QueryDecompose

import torch.optim as optim
import torch
from active import ActiveLearner, _to_datasets, print_eval_res, data_split_cv, save_eval_res
import cardnet
import os
from util import model_checkpoint, load_model, make_dir


def _summarize_eval_res(eval_res):
        if eval_res is None or len(eval_res) == 0:
                return 0.0, 0.0, 0
        res, loss, l1, _ = eval_res[0]
        cnt = len(res)
        avg_loss = loss / cnt if cnt > 0 else 0.0
        avg_l1 = l1 / cnt if cnt > 0 else 0.0
        return avg_loss, avg_l1, cnt


def _extract_q_error_records(eval_res, fold_idx):
        records = []
        if not eval_res:
                return records
        for dataset_idx, (res, _, _, _) in enumerate(eval_res):
                for sample_idx, (card_log, pred_log) in enumerate(res):
                        diff_log = float(pred_log - card_log)
                        q_error = float(2 ** abs(diff_log))
                        records.append({
                                "fold": fold_idx,
                                "dataset_index": dataset_idx,
                                "sample_index": sample_idx,
                                "card_log": float(card_log),
                                "pred_log": float(pred_log),
                                "diff_log": diff_log,
                                "q_error": q_error
                        })
        return records


def _summarize_q_errors(records):
	if not records:
		return None
	q_errors = np.array([rec["q_error"] for rec in records], dtype=np.float64)
	summary = {
		"count": int(q_errors.size),
		"mean": float(np.mean(q_errors)),
		"median": float(np.median(q_errors)),
		"p25": float(np.quantile(q_errors, 0.25)),
		"p75": float(np.quantile(q_errors, 0.75)),
		"min": float(np.min(q_errors)),
		"max": float(np.max(q_errors))
	}
	return summary


def _save_q_error_outputs(args, size, records, fold_summaries):
	if not records:
		return
	base_dir = os.path.join(args.save_res_dir, args.dataset, str(size))
	make_dir(base_dir)

	q_error_path = os.path.join(base_dir, "q_errors.csv")
	with open(q_error_path, "w", newline="") as csv_file:
		fieldnames = ["fold", "dataset_index", "sample_index", "card_log", "pred_log", "diff_log", "q_error"]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for record in records:
			writer.writerow(record)

	summary_path = os.path.join(base_dir, "q_error_summary.csv")
	with open(summary_path, "w", newline="") as csv_file:
		fieldnames = ["fold", "count", "mean", "median", "p25", "p75", "min", "max"]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for fold_summary in fold_summaries:
			writer.writerow(fold_summary)
		overall_summary = _summarize_q_errors(records)
		if overall_summary is not None:
			overall_summary = {"fold": "overall", **overall_summary}
			writer.writerow(overall_summary)


def pretrain_finetune_experiment(args):
	"""Run the requested pre-train + fine-tune experiment for each query size."""

	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	num_classes = args.max_classes

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed_all(args.seed)

	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args=args, all_subsets=all_subsets)
	QS.print_queryset_info()

	pretrain_queries, finetune_folds = QS.build_pretrain_finetune_splits(
		pretrain_ratio=args.pretrain_ratio,
		num_fold=args.num_fold,
		finetune_train_ratio=args.finetune_train_ratio,
		finetune_val_ratio=args.finetune_val_ratio,
		seed=args.seed)

	if len(pretrain_queries) == 0:
		raise RuntimeError("No queries available for pre-training. Please adjust the pretrain ratio or dataset.")

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	active_learner = ActiveLearner(args)

	pretrain_queries = list(pretrain_queries)
	pretrain_rng = random.Random(args.seed)
	pretrain_rng.shuffle(pretrain_queries)
	pretrain_val_count = int(len(pretrain_queries) * args.finetune_val_ratio)
	if pretrain_val_count >= len(pretrain_queries):
		pretrain_val_count = max(0, len(pretrain_queries) - 1)
	pretrain_val_queries = pretrain_queries[:pretrain_val_count]
	pretrain_train_queries = pretrain_queries[pretrain_val_count:]
	if len(pretrain_train_queries) == 0 and len(pretrain_val_queries) > 0:
		pretrain_train_queries = pretrain_val_queries
	pretrain_train_dataset = QueryDataset(pretrain_train_queries, num_classes=num_classes)
	pretrain_val_dataset = (
		QueryDataset(pretrain_val_queries, num_classes=num_classes)
		if len(pretrain_val_queries) > 0
		else pretrain_train_dataset
	)
	pretrain_datasets = [pretrain_train_dataset]
	pretrain_val_datasets = [pretrain_val_dataset]

	base_epochs = args.epochs
	summary_by_size = {}

	for size in sorted(finetune_folds.keys()):
		folds = finetune_folds[size]
		if size not in QS.all_sizes or len(QS.all_sizes[size]) == 0:
			continue

		print("\n" + "=" * 40)
		print("Processing queries with {} vertices".format(size))
		print("=" * 40)

		model = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
		model = model.to(args.device)
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)

		args.epochs = args.pretrain_epochs
		model, pretrain_time = active_learner.train(model=model, criterion=criterion, criterion_cal=criterion_cla,
			train_datasets=pretrain_datasets,
			val_datasets=pretrain_val_datasets,
			optimizer=optimizer,
			scheduler=scheduler,
			active=False)
		print("Pre-training time for size {}: {:.4f}s".format(size, pretrain_time))

		pretrained_state = copy.deepcopy(model.state_dict())

		fold_metrics = []
		size_records = []
		fold_summaries = []
		for fold_idx, (train_queries, val_queries, test_queries) in enumerate(folds):
			if len(train_queries) == 0 or len(test_queries) == 0:
				print("Skipping fold {} for size {} due to insufficient data.".format(fold_idx + 1, size))
				continue

			model_ft = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
			model_ft = model_ft.to(args.device)
			model_ft.load_state_dict(pretrained_state)
			optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
			scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=args.decay_factor)

			args.epochs = args.finetune_epochs
			train_datasets = _to_datasets([train_queries], num_classes)
			val_datasets = _to_datasets([val_queries], num_classes) if len(val_queries) > 0 else _to_datasets([[]], num_classes)
			test_datasets = _to_datasets([test_queries], num_classes)

			print("Fine-tuning fold {}/{} for size {} (train/val/test = {}/{}/{})".format(
				fold_idx + 1,
				args.num_fold,
				size,
				len(train_queries),
				len(val_queries),
				len(test_queries)))

			model_ft, _ = active_learner.train(model=model_ft, criterion=criterion, criterion_cal=criterion_cla,
				train_datasets=train_datasets,
				val_datasets=val_datasets,
				optimizer=optimizer_ft,
				scheduler=scheduler_ft,
				active=False)

			val_eval = active_learner.evaluate(model=model_ft, criterion=criterion, eval_datasets=val_datasets)
			test_eval = active_learner.evaluate(model=model_ft, criterion=criterion, eval_datasets=test_datasets)

			fold_summary = None
			test_records = _extract_q_error_records(test_eval, fold_idx + 1)
			if test_records:
				size_records.extend(test_records)
				fold_summary_values = _summarize_q_errors(test_records)
				if fold_summary_values is not None:
					fold_summary = {"fold": fold_idx + 1, **fold_summary_values}
					fold_summaries.append(fold_summary)

			val_loss, val_mae, val_cnt = _summarize_eval_res(val_eval)
			test_loss, test_mae, test_cnt = _summarize_eval_res(test_eval)

			test_q_error_mean = fold_summary["mean"] if fold_summary is not None else 0.0

			fold_metrics.append({
				"fold": fold_idx + 1,
				"train_count": len(train_queries),
				"val_count": val_cnt,
				"test_count": test_cnt,
				"val_loss": val_loss,
				"val_mae": val_mae,
				"test_loss": test_loss,
				"test_mae": test_mae,
				"test_q_error_mean": test_q_error_mean
			})

			print("Fold {} results - Val MAE: {:.4f}, Test MAE: {:.4f}".format(
				fold_idx + 1,
				val_mae,
				test_mae))
			if test_records and fold_summary is not None:
				print("Fold {} test q-error mean: {:.4f}, median: {:.4f}, p25: {:.4f}, p75: {:.4f}".format(
					fold_idx + 1,
					fold_summary["mean"],
					fold_summary["median"],
					fold_summary["p25"],
					fold_summary["p75"]))

		if len(fold_metrics) == 0:
			print("No valid folds generated for size {}.".format(size))
			continue

		if size_records:
			_save_q_error_outputs(args, size, size_records, fold_summaries)

		overall_q_summary = _summarize_q_errors(size_records)
		if overall_q_summary is not None:
			print("Summary for size {} - Test q-error mean: {:.4f}, median: {:.4f}, p25: {:.4f}, p75: {:.4f}".format(
				size,
				overall_q_summary["mean"],
				overall_q_summary["median"],
				overall_q_summary["p25"],
				overall_q_summary["p75"]))

		val_maes = [m["val_mae"] for m in fold_metrics if m["val_count"] > 0]
		test_maes = [m["test_mae"] for m in fold_metrics if m["test_count"] > 0]
		summary_by_size[size] = {
			"folds": fold_metrics,
			"val_mae_mean": float(np.mean(val_maes)) if len(val_maes) > 0 else 0.0,
			"val_mae_std": float(np.std(val_maes)) if len(val_maes) > 0 else 0.0,
			"test_mae_mean": float(np.mean(test_maes)) if len(test_maes) > 0 else 0.0,
			"test_mae_std": float(np.std(test_maes)) if len(test_maes) > 0 else 0.0,
			"test_q_error_mean": overall_q_summary["mean"] if overall_q_summary is not None else 0.0,
			"test_q_error_median": overall_q_summary["median"] if overall_q_summary is not None else 0.0,
			"test_q_error_p25": overall_q_summary["p25"] if overall_q_summary is not None else 0.0,
			"test_q_error_p75": overall_q_summary["p75"] if overall_q_summary is not None else 0.0
		}

		print("Summary for size {} - Test MAE Mean: {:.4f}, Std: {:.4f}".format(
			size,
			summary_by_size[size]["test_mae_mean"],
			summary_by_size[size]["test_mae_std"]))

	args.epochs = base_epochs

	print("\nFinal summary by query size:")
	if not summary_by_size:
		print("No fine-tuning results were generated.")
	for size in sorted(summary_by_size.keys()):
		summary = summary_by_size[size]
		print("Size {} -> Val MAE Mean {:.4f} (Std {:.4f}), Test MAE Mean {:.4f} (Std {:.4f})".format(
			size,
			summary["val_mae_mean"],
			summary["val_mae_std"],
			summary["test_mae_mean"],
			summary["test_mae_std"]))
		print("           Test q-error Mean {:.4f}, Median {:.4f}, P25 {:.4f}, P75 {:.4f}".format(
			summary["test_q_error_mean"],
			summary["test_q_error_median"],
			summary["test_q_error_p25"],
			summary["test_q_error_p75"]))


def _summarize_eval_res(eval_res):
	if eval_res is None or len(eval_res) == 0:
		return 0.0, 0.0, 0
	res, loss, l1, _ = eval_res[0]
	cnt = len(res)
	avg_loss = loss / cnt if cnt > 0 else 0.0
	avg_l1 = l1 / cnt if cnt > 0 else 0.0
	return avg_loss, avg_l1, cnt


def pretrain_finetune_experiment(args):
	"""Run the requested pre-train + fine-tune experiment for each query size."""

	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	num_classes = args.max_classes

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed_all(args.seed)

	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args=args, all_subsets=all_subsets)
	QS.print_queryset_info()

	pretrain_queries, finetune_folds = QS.build_pretrain_finetune_splits(
		pretrain_ratio=args.pretrain_ratio,
		num_fold=args.num_fold,
		finetune_train_ratio=args.finetune_train_ratio,
		finetune_val_ratio=args.finetune_val_ratio,
		seed=args.seed)

	if len(pretrain_queries) == 0:
		raise RuntimeError("No queries available for pre-training. Please adjust the pretrain ratio or dataset.")

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	active_learner = ActiveLearner(args)

	pretrain_queries = list(pretrain_queries)
	pretrain_rng = random.Random(args.seed)
	pretrain_rng.shuffle(pretrain_queries)
	pretrain_val_count = int(len(pretrain_queries) * args.finetune_val_ratio)
	if pretrain_val_count >= len(pretrain_queries):
		pretrain_val_count = max(0, len(pretrain_queries) - 1)
	pretrain_val_queries = pretrain_queries[:pretrain_val_count]
	pretrain_train_queries = pretrain_queries[pretrain_val_count:]
	if len(pretrain_train_queries) == 0 and len(pretrain_val_queries) > 0:
		pretrain_train_queries = pretrain_val_queries
	pretrain_train_dataset = QueryDataset(pretrain_train_queries, num_classes=num_classes)
	pretrain_val_dataset = (
		QueryDataset(pretrain_val_queries, num_classes=num_classes)
		if len(pretrain_val_queries) > 0
		else pretrain_train_dataset
	)
	pretrain_datasets = [pretrain_train_dataset]
	pretrain_val_datasets = [pretrain_val_dataset]

	base_epochs = args.epochs
	summary_by_size = {}

	for size in sorted(finetune_folds.keys()):
		folds = finetune_folds[size]
		if size not in QS.all_sizes or len(QS.all_sizes[size]) == 0:
			continue

		print("\n" + "=" * 40)
		print("Processing queries with {} vertices".format(size))
		print("=" * 40)

		model = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
		model = model.to(args.device)
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)

		args.epochs = args.pretrain_epochs
		model, pretrain_time = active_learner.train(model=model, criterion=criterion, criterion_cal=criterion_cla,
			train_datasets=pretrain_datasets,
			val_datasets=pretrain_val_datasets,
			optimizer=optimizer,
			scheduler=scheduler,
			active=False)
		print("Pre-training time for size {}: {:.4f}s".format(size, pretrain_time))

		pretrained_state = copy.deepcopy(model.state_dict())

		fold_metrics = []
		for fold_idx, (train_queries, val_queries, test_queries) in enumerate(folds):
			if len(train_queries) == 0 or len(test_queries) == 0:
				print("Skipping fold {} for size {} due to insufficient data.".format(fold_idx + 1, size))
				continue

			model_ft = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
			model_ft = model_ft.to(args.device)
			model_ft.load_state_dict(pretrained_state)
			optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
			scheduler_ft = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=args.decay_factor)

			args.epochs = args.finetune_epochs
			train_datasets = _to_datasets([train_queries], num_classes)
			val_datasets = _to_datasets([val_queries], num_classes) if len(val_queries) > 0 else _to_datasets([[]], num_classes)
			test_datasets = _to_datasets([test_queries], num_classes)

			print("Fine-tuning fold {}/{} for size {} (train/val/test = {}/{}/{})".format(
				fold_idx + 1,
				args.num_fold,
				size,
				len(train_queries),
				len(val_queries),
				len(test_queries)))

			model_ft, _ = active_learner.train(model=model_ft, criterion=criterion, criterion_cal=criterion_cla,
				train_datasets=train_datasets,
				val_datasets=val_datasets,
				optimizer=optimizer_ft,
				scheduler=scheduler_ft,
				active=False)

			val_eval = active_learner.evaluate(model=model_ft, criterion=criterion, eval_datasets=val_datasets)
			test_eval = active_learner.evaluate(model=model_ft, criterion=criterion, eval_datasets=test_datasets)

			val_loss, val_mae, val_cnt = _summarize_eval_res(val_eval)
			test_loss, test_mae, test_cnt = _summarize_eval_res(test_eval)

			fold_metrics.append({
				"fold": fold_idx + 1,
				"train_count": len(train_queries),
				"val_count": val_cnt,
				"test_count": test_cnt,
				"val_loss": val_loss,
				"val_mae": val_mae,
				"test_loss": test_loss,
				"test_mae": test_mae
			})

			print("Fold {} results - Val MAE: {:.4f}, Test MAE: {:.4f}".format(
				fold_idx + 1,
				val_mae,
				test_mae))

		if len(fold_metrics) == 0:
			print("No valid folds generated for size {}.".format(size))
			continue

		val_maes = [m["val_mae"] for m in fold_metrics if m["val_count"] > 0]
		test_maes = [m["test_mae"] for m in fold_metrics if m["test_count"] > 0]

		summary_by_size[size] = {
			"folds": fold_metrics,
			"val_mae_mean": float(np.mean(val_maes)) if len(val_maes) > 0 else 0.0,
			"val_mae_std": float(np.std(val_maes)) if len(val_maes) > 0 else 0.0,
			"test_mae_mean": float(np.mean(test_maes)) if len(test_maes) > 0 else 0.0,
			"test_mae_std": float(np.std(test_maes)) if len(test_maes) > 0 else 0.0
		}

		print("Summary for size {} - Test MAE Mean: {:.4f}, Std: {:.4f}".format(
			size,
			summary_by_size[size]["test_mae_mean"],
			summary_by_size[size]["test_mae_std"]))

	args.epochs = base_epochs

	print("\nFinal summary by query size:")
	if not summary_by_size:
		print("No fine-tuning results were generated.")
	for size in sorted(summary_by_size.keys()):
		summary = summary_by_size[size]
		print("Size {} -> Val MAE Mean {:.4f} (Std {:.4f}), Test MAE Mean {:.4f} (Std {:.4f})".format(
			size,
			summary["val_mae_mean"],
			summary["val_mae_std"],
			summary["test_mae_mean"],
			summary["test_mae_std"]))


def main(args):
	"""
	Entrance of train/test/active learning
	"""
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	data_dir = args.data_dir
	num_classes = args.max_classes

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor


	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args= args, all_subsets=all_subsets)

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()

	train_sets, val_sets, test_sets, all_train_sets = QS.train_sets, QS.val_sets, QS.test_sets, QS.all_train_sets
	train_datasets = _to_datasets(train_sets, num_classes) if args.cumulative else _to_datasets(all_train_sets, num_classes)
	val_datasets, test_datasets, = _to_datasets(val_sets, num_classes), _to_datasets(test_sets, num_classes)

	model = cardnet.CardNet(args, num_node_feat= num_node_feat, num_edge_feat = num_edge_feat)
	print(model)
	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

	active_learner = ActiveLearner(args)
	if args.mode == "train":
		print("start active learning ...")
		#load_model(args=args, model=model, device=args.device, optimizer=optimizer)
		active_learner.active_train(model=model, criterion=criterion, criterion_cla=criterion_cla,
								train_datasets=train_datasets, val_datasets=val_datasets, test_datasets=test_datasets,
								optimizer=optimizer, scheduler=scheduler, pretrain=True)
		model_checkpoint(args=args, model=model, optimizer=optimizer, scheduler=scheduler)

	elif args.mode == "pretrain":
		active_learner.active_train(model=model, criterion=criterion, criterion_cla=criterion_cla,
									train_datasets=train_datasets, val_datasets=val_datasets,
									test_datasets=test_datasets,
									optimizer=optimizer, scheduler=scheduler, pretrain=True)
		model_checkpoint(args=args, model=model, optimizer=optimizer, scheduler=scheduler)

	elif args.mode == "test":
		print("loading model ...")
		load_model(args = args, model=model, device=args.device, optimizer= optimizer)
		print("make prediction ...")
		active_learner.evaluate(model=model, criterion= criterion, eval_datasets=val_datasets, print_res=True)

def ensemble_learn(args):
	"""
	Entrance of Ensemble active learning
	"""
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	data_dir = args.data_dir

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor


	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args= args, all_subsets=all_subsets)

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()

	train_sets, val_sets, test_sets, all_train_sets = QS.train_sets, QS.val_sets, QS.test_sets, QS.all_train_sets
	train_datasets = _to_datasets(train_sets) if args.cumulative else _to_datasets(all_train_sets)
	val_datasets, test_datasets, = _to_datasets(val_sets), _to_datasets(test_sets)

	models = []
	for _ in range(args.ensemble_num):
		models.append(cardnet.CardNet(args, num_node_feat= num_node_feat, num_edge_feat = num_edge_feat))

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	optimizers = [ optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) for model in models]
	schedulers = [ optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor) for optimizer in optimizers]

	active_learner = ActiveLearner(args)
	active_learner.ensemble_active_train(models, criterion,criterion_cla,
										 train_datasets, val_datasets, test_datasets, optimizers, schedulers, pretrain=True)



def cross_validate(args):
	"""
	Entrance of cross validation, without active learning
	"""
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	num_classes = args.max_classes

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor

	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args=args, all_subsets=all_subsets)
	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()
	all_sizes = QS.all_sizes # {size -> (graphs, card)}
	all_fold_train_sets, all_fold_val_sets = data_split_cv(all_sizes, num_fold=args.num_fold)

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	active_learner = ActiveLearner(args)
	all_fold_val_res = None
	i = 0
	total_elapse_time = 0.0
	for train_sets, val_sets in zip(all_fold_train_sets, all_fold_val_sets):
		i += 1
		print("start the {}/{} fold training ...".format(i, args.num_fold))
		train_datasets, val_datasets = _to_datasets([train_sets], num_classes), _to_datasets(val_sets, num_classes)
		model = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
		print(model)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)
		_, fold_elapse_time = active_learner.train(model=model, criterion=criterion, criterion_cal=criterion_cla,
									train_datasets=train_datasets, val_datasets=val_datasets,
									optimizer=optimizer, scheduler=scheduler, active=False)
		total_elapse_time += fold_elapse_time
		fold_eval_res = active_learner.evaluate(model=model, criterion=criterion, eval_datasets=val_datasets)
		# merge the result of the evaluation result of each fold
		if all_fold_val_res is None:
			all_fold_val_res = fold_eval_res
		else:
			tmp_all_fold_val_res = []
			for all_res, fold_res in zip(all_fold_val_res, fold_eval_res):
				tmp_res = all_res[0] + fold_res[0]
				tmp_loss = all_res[1] + fold_res[1]
				tmp_l1 = all_res[2] + fold_res[2]
				tmp_elapse_time = all_res[3] + fold_res[3]
				tmp_all_fold_val_res.append((tmp_res, tmp_loss, tmp_l1, tmp_elapse_time))
			all_fold_val_res = tmp_all_fold_val_res
	print("the average training time: {:.4f}(s)".format(total_elapse_time / args.num_fold))
	print("the total evaluation result:")
	error_median = print_eval_res(all_fold_val_res, print_details=False)
	print("error_median={}".format(0 - error_median))
	save_eval_res(args, sorted(all_sizes.keys()), all_fold_val_res, args.save_res_dir)


if __name__ == "__main__":
	parser = ArgumentParser("LSS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	# Model Settings (ONLY FOR CardNet MODEL)
	parser.add_argument("--num_layers", default=3, type=int,
						help="number of convolutional layers")
	parser.add_argument("--model_type", default="NNGINConcat", type=str,
						help="GNN layer type") # GIN, GINE, GAT, NN, GCN, SAGE, NNGIN, NNGINConcat
	parser.add_argument("--embed_type", default="freq", type=str,
						help="the node feature encoding type") # freq, n2v, prone, n2v_concat, prone_concat, nrp are tested
	parser.add_argument("--edge_embed_type", default="freq", type=str,
						help="the edge feature encoding type")
	parser.add_argument("--num_g_hid", default=128, type=int,
						help="hidden dim for transforming nodes for intermediate GNN layer")
	parser.add_argument("--num_e_hid", default=32, type=int,
						help="hidden dim for transforming edges for intermediate GNN layer")
	parser.add_argument("--out_g_ch", default=128, type=int,
						help="number of output dimension of the final GNN layer")
	parser.add_argument("--num_expert", default=64, type=int,
						help="hyper-parameter for the attention layer")
	parser.add_argument("--num_att_hid", default=64, type=int,
						help="hyper-parameter for the attention layer")
	parser.add_argument("--num_mlp_hid", default=128, type=int,
						help="number of hidden units of MLP")
	parser.add_argument('--pool_type', type=str, default="att",  # att, mean, sum, max
						help='shards pooling layer type')
	parser.add_argument('--dropout', type=float, default=0.2,
						help='Dropout rate (1 - keep probability).')
	# Training settings
	parser.add_argument("--cumulative", default=False, type=bool,
					help='Whether or not to enable cumulative learning')
	parser.add_argument("--num_fold", default=5, type=int,
					help="number of fold for cross validation")
	parser.add_argument("--epochs", default=80, type=int)
	parser.add_argument("--pretrain_epochs", default=30, type=int,
					help="number of epochs during the pre-training stage")
	parser.add_argument("--finetune_epochs", default=50, type=int,
					help="number of epochs during the fine-tuning stage")
	parser.add_argument("--batch_size", default= 2, type=int)
	parser.add_argument("--learning_rate", default= 1e-4, type=float)
	parser.add_argument('--weight_decay', type=float, default=5e-4,
					help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--decay_factor', type=float, default=0.1,
					help='decay rate of (gamma).')
	parser.add_argument('--decay_patience', type=int, default=50,
					help='num of epochs for one lr decay.')
	parser.add_argument('--weight_exp', type=float, default=1.0,
					help='loss weight exp factor.')
	parser.add_argument('--no-cuda', action='store_true', default=False,
					help='Disables CUDA training.')
	parser.add_argument('--num_workers', type = int, default= 16,
					help='number of workers for Dataset.')
	parser.add_argument("--pretrain_ratio", type=float, default=0.2,
					help="ratio of each size bucket used for shared pre-training")
	parser.add_argument("--finetune_train_ratio", type=float, default=0.8,
					help="training ratio within the fine-tuning splits")
	parser.add_argument("--finetune_val_ratio", type=float, default=0.1,
					help="validation ratio within the fine-tuning splits")
	parser.add_argument("--seed", type=int, default=1,
					help="random seed for split reproducibility")
	# Classification task settings
	parser.add_argument("--multi_task", default=True, type=bool,
						help="enable/disable card classification task.")
	parser.add_argument("--max_classes", default=10, type=int,
						help="number classes for the card classification task.")
	parser.add_argument('--coeff', type=float, default=0.5,
						help='coefficient for the classification loss.')
	# Active Learner settings
	parser.add_argument("--uncertainty", default="consist", type=str,
						help="The uncertainty type") # entropy, margin, confident, consist, random are tested
	parser.add_argument("--biased_sample", default=True, type=bool,
						help="Enable Biased sampling for test set selection")
	parser.add_argument('--active_iters', type=int, default=2,
						help='Num of iterators of active learning.')
	parser.add_argument('--budget', type=int, default=50,
						help='Selected Queries budget Per Iteration.')
	parser.add_argument('--active_epochs', type=int, default=50,
						help='Training Epochs for per iteration active learner.')
	parser.add_argument('--ensemble_num', type=int, default=5,
						help='number of ensemble models for active learning.')
	# Input and Output directory
	parser.add_argument("--dataset", type=str, default="aids")  # aids, wordnet, yeast, hprd, youtube, eu2005 are tested
	parser.add_argument("--full_data_dir", type=str, default="./data/")
	parser.add_argument("--save_res_dir", type=str, default="./result/")
	parser.add_argument("--model_file", type=str, default="aids_homo.pth")
	parser.add_argument("--model_save_dir", type=str, default="./models")

	# Other parameters
	parser.add_argument("--matching", default="homo", type=str,
						help="The subgraph matching mode")
	parser.add_argument('--k', type=int, default=3,
						help='decompose hop number.')
	parser.add_argument("--verbose", default=True, type=bool)
	parser.add_argument("--mode", default="cross_val", type=str,
						help="The running mode") # train (train & test) or test (only test) or pretrain or ensemble or cross_val or pretrain_finetune
	args = parser.parse_args()

	# set the hardware parameter
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	# set the input dir
	args.queryset_iso_dir = os.path.join(args.full_data_dir, "queryset")
	args.queryset_homo_dir = os.path.join(args.full_data_dir, "queryset_homo")
	args.true_iso_dir = os.path.join(args.full_data_dir, "true_cardinality")
	args.true_homo_dir = os.path.join(args.full_data_dir, "true_homo")
	args.data_dir = os.path.join(args.full_data_dir, "dataset")
	args.prone_feat_dir = os.path.join(args.full_data_dir, "prone")
	args.n2v_feat_dir = os.path.join(args.full_data_dir, "n2v")
	args.nrp_feat_dir = os.path.join(args.full_data_dir, "nrp")

	args.embed_feat_dir = args.n2v_feat_dir if args.embed_type == "n2v" or args.embed_type == "n2v_concat" else \
		args.prone_feat_dir
	if args.embed_type == "nrp":
		args.embed_feat_dir = args.nrp_feat_dir
	args.active_iters = 0 if args.mode in {"pretrain", "pretrain_finetune"} else args.active_iters
	args.queryset_dir = args.queryset_homo_dir if args.matching == "homo" else  args.queryset_iso_dir
	args.true_card_dir = args.true_homo_dir if args.matching == "homo" else args.true_iso_dir


	if args.verbose:
		print(args)
	if args.mode == "cross_val":
		cross_validate(args)
	elif args.mode == "ensemble":
		ensemble_learn(args)
	elif args.mode == "pretrain_finetune":
		pretrain_finetune_experiment(args)
	else: # train/test/pre-train
		main(args)
