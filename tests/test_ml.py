import io

import pytest
import torch
from PIL import Image

from grounding.ml import Grounder, Tokenizer, grounding_loss, preprocess_image, box_iou_giou, make_batch
from grounding.inference import calculate_metrics
from grounding.training import default_config


@pytest.fixture
def small_model():
    torch.set_num_threads(2)
    config = {**default_config(), "image_size": 32, "width": 8, "text_dim": 16}
    return Grounder(7, 6, config)


def test_boxes_are_valid_even_with_saturated_logits(small_model):
    images = torch.randn(2, 3, 32, 32)
    tokens, lengths = torch.ones(2, 4, dtype=torch.long), torch.tensor([4, 2])
    for values in ([1000, 1000, -1000, -1000], [-1000, -1000, 1000, 1000], [0, 0, 0, 0]):
        with torch.no_grad():
            small_model.box_head.weight.zero_()
            small_model.box_head.bias.copy_(torch.tensor(values))
        boxes = small_model(images, tokens, lengths)["bbox"]
        assert torch.all(boxes[:, :2] < boxes[:, 2:])
        assert torch.all((boxes >= 0) & (boxes <= 1))


@pytest.mark.parametrize("present", [[0, 0], [1, 0], [1, 1]])
def test_losses_mask_absent_targets_and_remain_finite(small_model, present):
    output = small_model(torch.randn(2, 3, 32, 32), torch.ones(2, 4, dtype=torch.long), torch.tensor([4, 4]))
    output["bbox"].retain_grad()
    output["class_logits"].retain_grad()
    boxes = torch.tensor([[.1, .2, .4, .5], [.2, .3, .5, .6]])
    labels = torch.tensor([2, 3])
    for i, positive in enumerate(present):
        if not positive:
            boxes[i] = float("nan")
            labels[i] = -999
    loss = grounding_loss(output, {"present": torch.tensor(present), "boxes": boxes, "classes": labels})
    assert all(torch.isfinite(value) for value in loss.values())
    loss["total"].backward()
    for i, positive in enumerate(present):
        if not positive:
            assert torch.count_nonzero(output["bbox"].grad[i]) == 0
            assert torch.count_nonzero(output["class_logits"].grad[i]) == 0
    if not any(present):
        assert loss["l1"].item() == loss["giou"].item() == loss["classification"].item() == 0


def test_known_iou():
    box = torch.tensor([[0., 0., 1., 1.]])
    target = torch.tensor([[0., 0., .5, .5]])
    iou, giou = box_iou_giou(box, target)
    assert iou.item() == pytest.approx(.25)
    assert giou.item() == pytest.approx(.25)


def test_tokenizer_train_only_unknown_and_truncation():
    tokenizer = Tokenizer.build(["Find blue button"], max_tokens=4)
    assert "validationword" not in tokenizer.vocabulary
    assert tokenizer.encode("validationword")[0][0] == 1
    assert tokenizer.encode("find blue button blue button")[1] == 4
    assert Tokenizer.from_dict(tokenizer.to_dict()).encode("blue") == tokenizer.encode("blue")


def test_prediction_and_training_preprocessing_identical(tmp_path):
    image = Image.new("RGBA", (71, 39), (20, 50, 100, 128))
    path = tmp_path / "image.png"
    image.save(path)
    config = {**default_config(), "image_size": 32}
    tokenizer = Tokenizer.build(["find button"])
    record = {"image_path": "image.png", "instruction": "find button", "target_present": False}
    batch = make_batch([record], tokenizer, config, tmp_path, torch.device("cpu"))
    with Image.open(path) as opened:
        predicted = preprocess_image(opened, config["image_size"])
    assert torch.equal(batch["images"][0], predicted)
    assert batch["images"].shape == (1, 3, 32, 32)


def test_abstention_counts_as_grounding_failure():
    record = {"image_path": "image.png", "target_present": True, "bbox": [.1, .1, .5, .5], "class_id": 0}
    correct = {"target_present": True, "raw_bbox": record["bbox"], "raw_class_id": 0, "latency_ms": 2}
    abstain = {**correct, "target_present": False}
    absent = {"image_path": "absent.png", "target_present": False}
    metrics, _ = calculate_metrics([record, record, absent], [correct, abstain, correct])
    assert metrics["grounding_success"] == .5
    assert metrics["selective_correctness"] == .5
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate_absent"] == 1
    assert metrics["positive_abstentions"] == 1


def test_absent_only_metrics_have_undefined_positive_denominators():
    metrics, _ = calculate_metrics([{"image_path": "x.png", "target_present": False}],
                                    [{"target_present": False, "raw_bbox": [0, 0, 1, 1], "raw_class_id": 0, "latency_ms": 0}])
    assert metrics["grounding_success"] is None
    assert metrics["presence_precision"] is None
    assert metrics["presence_accuracy"] == 1
