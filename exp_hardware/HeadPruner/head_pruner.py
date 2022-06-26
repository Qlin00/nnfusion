import os
import numpy as np
import torch
from scipy.spatial import distance


def att_mask_list2str(att_mask: list):
    if not sum([len(head) for head in att_mask]):
        return "empty"
    res = []
    for ii, jj in enumerate(att_mask):
        if not jj:
            continue
        layer = "encoder"
        idx = ii
        tmp = f"{layer}{idx + 1}:{','.join([str(i + 1) for i in sorted(jj)])}"
        res.append(tmp)
    return "-".join(res)


def str2att_mask_list(att_mask_str: str, layer_num: int = 12):
    res = [[] for _ in range(layer_num)]
    if att_mask_str in ["empty", ""]:
        return res
    for x in att_mask_str.split("-"):
        layer, heads = x.replace("encoder", "").split(":")
        layer = int(layer) - 1
        idx = [int(ii) - 1 for ii in heads.split("+")]
        res[layer] = idx
    return res


class HeadPruner(object):
    def __init__(
        self,
        model,
        method: str,
        valid_dataset,
        test_dataset,
        config: dict,
        params: dict,
    ):
        self.model = model
        self.method = method
        self.params = params
        self.dataset = {"valid": valid_dataset, "test": test_dataset}
        if params.get("use_baseline", False):
            self.get_baseline_result(config)
        os.makedirs(self.params.get("output_dir", "log/"), exist_ok=True)

    def predict(self, model, dataset, config: dict):
        """return pred_prob, pred_labels, metrics"""
        raise NotImplementedError("Implement in subclass")

    def mask_head(self, model, heads: list, config: dict):
        """return model"""
        raise NotImplementedError("Implement in subclass")

    def get_baseline_result(self, config: dict):
        valid_baseline = self.predict(self.model, self.dataset["valid"], config)
        test_baseline = self.predict(self.model, self.dataset["test"], config)
        self.baseline = {"valid": valid_baseline, "test": test_baseline}

    def evaluate_head_masked(self, heads: list, train_mode: str, config: dict):
        model = self.mask_head(self.model, heads, config)
        dataset = self.dataset[train_mode]
        if self.params.get("use_baseline", False):
            baseline = self.baseline[train_mode]
        pred_prob, pred_labels, metrics = self.predict(model, dataset, config)
        if self.params.get("use_baseline", False):
            pl, ll = self.compare_with_baseline(pred_prob, pred_labels, baseline)
            metrics["pl"] = pl
            metrics["ll"] = ll
        self.log_result(heads, metrics, train_mode, config)
        return metrics

    def compare_with_baseline(self, pred_prob: list, pred_labels: list, baseline: list):
        """return probability loyalty, label loyalty"""
        dj = distance.jensenshannon(baseline[0], pred_prob)
        pl = (1 - np.sqrt(dj)).mean()
        ll = accuracy_score(baseline[1], pred_labels)
        return pl, ll

    def log_result(self, heads: list, metrics: dict, train_mode: str, config: dict):
        if config.get("local_rank", -1) not in [-1, 0]:
            return
        if train_mode == "valid":
            output_dir = "{}{}".format(
                config.get("output_dir", "./log/"), config.get("mask_idx", 0)
            )
        else:
            output_dir = self.params.get("output_dir", "./log/")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "log.txt")
        log_str = "{},{},{},{}{}\n".format(
            config.get("mask_idx", 0),
            sum([len(ii) for ii in heads]),
            att_mask_list2str(heads).replace(",", "+"),
            ",".join(
                [
                    f"{metrics[metric_name] * 100:.3f}"
                    for metric_name in self.params.get(
                        "metrics_list",
                        [self.params.get("metrics_name", "predict_f1")],
                    )
                ]
            ),
            f",{metrics['pl'] * 100:.3f},{metrics['ll'] * 100:.3f}"
            if self.params.get("use_baseline", False)
            else "",
        )
        with open(output_path, "a") as f:
            f.write(log_str)

    def beam_search_union(self, config: dict):
        def evaulate_mask(layer_id: int, head_id: int, beam: tuple):
            tmp_heads = [ii.copy() for ii in heads]
            for beam_layer, beam_head in enumerate(beam):
                if beam_head != -1:
                    tmp_heads[beam_layer].append(beam_head)
            tmp_heads[layer_id].append(head_id)
            metrics = self.evaluate_head_masked(tmp_heads, "valid", config)
            tmp_beams[tuple(list(beam) + [head_id])] = metrics[metrics_name]

        config["output_dir"] = os.path.join(
            self.params.get("output_dir", "./log/"), "beam_search_union"
        )
        layer_num, head_num = self.params.get("layer_num", 12), self.params.get(
            "head_num", 12
        )
        stop_threshold = self.params.get("stop_threshold", 0)
        heads = config.get("heads", [[] for ii in range(layer_num)])
        metrics_name = self.params.get("metrics_name", "predict_f1")
        prev_metric = self.evaluate_head_masked(heads, "valid", config)[metrics_name]
        self.evaluate_head_masked(heads, "test", config)

        while True:
            beams = {tuple(): prev_metric}
            config["mask_idx"] = config.get("mask_idx", 0) + 1
            for layer_id in range(layer_num):
                if layer_id < len(list(beams.keys())[0]):
                    continue
                tmp_beams = {tuple(list(i) + [-1]): j for i, j in beams.items()}
                for beam in beams.keys():
                    for head_id in range(head_num):
                        if (
                            head_id not in heads[layer_id]
                            and len(heads[layer_id]) < head_num - 1
                        ):
                            evaulate_mask(layer_id, head_id, beam)
                beams = {
                    beam: metric
                    for beam, metric in sorted(tmp_beams.items(), key=lambda i: -i[1])[
                        :3
                    ]
                }
                print("beams", beams)

            for beam, metric in beams.items():
                for layer_id, head_id in enumerate(beam):
                    if (
                        head_id != -1
                        and head_id not in heads[layer_id]
                        and len(heads[layer_id]) < head_num - 1
                    ):
                        heads[layer_id].append(head_id)
            prev_metric = self.evaluate_head_masked(heads, "valid", config)[
                metrics_name
            ]
            test_metric = self.evaluate_head_masked(heads, "test", config)[metrics_name]
            if test_metric <= stop_threshold:
                break
            config["mask_idx"] += 1
