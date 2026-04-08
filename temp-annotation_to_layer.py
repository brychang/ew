import ast
import copy
import json
import re

import pandas as pd
from caveclient import CAVEclient


def df_to_annotations(
    df,
    pointA="pointA",
    pointB="pointB",
    idcol="id",
    description="description",
    typecol=None,
    annotation_type="line",
):

    def _parse_point(value):
        if isinstance(value, (list, tuple)):
            coords = [float(v) for v in value]
        else:
            cleaned = re.sub(r"np\.\w+\(([^()]*)\)", r"\1", str(value))
            try:
                parsed = ast.literal_eval(cleaned)
                if isinstance(parsed, (list, tuple)):
                    coords = [float(v) for v in parsed]
                else:
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
                    coords = [float(v) for v in nums]
            except (SyntaxError, ValueError):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
                coords = [float(v) for v in nums]

        if len(coords) < 3:
            raise ValueError(f"Point must have at least 3 numeric values, got: {value}")

        return coords[:3]

    required_cols = [pointA, pointB, idcol, description]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    select_cols = [pointA, pointB, idcol, description]
    if typecol is not None:
        if typecol not in df.columns:
            raise ValueError(f"typecol {typecol!r} not found in df columns")
        select_cols.append(typecol)

    df = df[select_cols].copy()

    df = df.rename(
        columns={
            pointA: "pointA",  # rename
            pointB: "pointB",
            idcol: "id",
            description: "description",
            **({typecol: "type"} if typecol is not None else {}),
        }
    )

    if typecol is None:
        df["type"] = annotation_type

    df["pointA"] = df["pointA"].map(_parse_point)
    df["pointB"] = df["pointB"].map(_parse_point)

    df = df.astype({"id": str, "description": str, "type": str})  # make sure right type

    annot_dict = df.to_dict(orient="records")  # transform to dict

    return annot_dict


def annotations_to_layer(
    layer_template=None, annot_dict=None, name="", color="#ffffff", visible=False
):

    if layer_template is None:
        layer_template = {"type": "annotation", "annotations": []}
    if annot_dict is None:
        annot_dict = []

    layer = copy.deepcopy(layer_template)
    layer["name"] = name
    layer["annotations"] = annot_dict
    layer["annotationColor"] = color
    layer["visible"] = visible

    return layer


def df_to_layer(
    df,
    name,
    color,
    layer_template=None,
    visible=False,
    pointA="pointA",
    pointB="pointB",
    idcol="id",
    description="description",
    typecol=None,
    annotation_type="line",
):

    annot_dict = df_to_annotations(
        df,
        pointA=pointA,
        pointB=pointB,
        idcol=idcol,
        description=description,
        typecol=typecol,
        annotation_type=annotation_type,
    )

    return annotations_to_layer(
        layer_template=layer_template,
        annot_dict=annot_dict,
        name=name,
        color=color,
        visible=visible,
    )


def segments_to_layer(layer_template=None, segments_list=None, name="", visible=False):

    if layer_template is None:
        layer_template = {"type": "segmentation", "segments": []}
    if segments_list is None:
        segments_list = []

    layer = copy.deepcopy(layer_template)
    layer["name"] = name
    layer["segments"] = segments_list
    layer["visible"] = visible

    return layer


def load_layer_template_from_state_json(
    state_json_path, layer_name=None, layer_type=None
):

    with open(state_json_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    layers = state.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("State JSON must contain a list at key 'layers'.")

    for layer in layers:
        if layer_name is not None and layer.get("name") != layer_name:
            continue
        if layer_type is not None and layer.get("type") != layer_type:
            continue
        return copy.deepcopy(layer)

    filters = []
    if layer_name is not None:
        filters.append(f"name={layer_name!r}")
    if layer_type is not None:
        filters.append(f"type={layer_type!r}")
    filt_txt = ", ".join(filters) if filters else "(no filter)"
    raise ValueError(f"No layer found in state JSON matching: {filt_txt}")


def load_state_json(state_json_path):

    with open(state_json_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    if not isinstance(state, dict):
        raise ValueError("State JSON must be an object/dict.")

    layers = state.get("layers")
    if layers is None:
        state["layers"] = []
    elif not isinstance(layers, list):
        raise ValueError("State JSON key 'layers' must be a list.")

    return state


def add_layer_to_state(state, layer, replace_existing_name=True):

    out_state = copy.deepcopy(state)
    out_layers = out_state.get("layers", [])

    if replace_existing_name and layer.get("name"):
        out_layers = [ly for ly in out_layers if ly.get("name") != layer["name"]]

    out_layers.append(layer)
    out_state["layers"] = out_layers

    return out_state


def make_link(data):

    client = CAVEclient("stroeh_mouse_retina")

    link_template = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/"
    new_id = client.state.upload_state_json(data)
    new_link = f"{link_template}{new_id}"

    print(new_link)
    return new_link


if __name__ == "__main__":
    df = pd.read_csv("data/OFF SAC_720575940567969903.csv", index_col=0)

    layer = df_to_layer(
        df,
        name="my_annotations",
        color="#ff0000",
        visible=True,
        idcol="rb_id",
        description="partner_type",
        annotation_type="line",
    )

    base_state = load_state_json("data/state.json")
    state_with_annotations = add_layer_to_state(base_state, layer)
    make_link(state_with_annotations)
