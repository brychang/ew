# %%
import copy
import json

import pandas as pd
from caveclient import CAVEclient


# %%
def df_to_point_annotations(
    df,
    xcol="centroid_x",
    ycol="centroid_y",
    zcol="centroid_z",
    idcol="source_row",
    descriptioncol="Cell Type",
):

    required_cols = [xcol, ycol, zcol, idcol, descriptioncol]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    out = df[[xcol, ycol, zcol, idcol, descriptioncol]].copy()
    out = out.rename(
        columns={
            xcol: "x",
            ycol: "y",
            zcol: "z",
            idcol: "id",
            descriptioncol: "description",
        }
    )

    out["x"] = pd.to_numeric(out["x"], errors="raise")
    out["y"] = pd.to_numeric(out["y"], errors="raise")
    out["z"] = pd.to_numeric(out["z"], errors="raise")

    out["point"] = out[["x", "y", "z"]].values.tolist()
    out["id"] = out["id"].astype(str)
    out["description"] = out["description"].astype(str)
    out["type"] = "point"

    annot_df = out[["point", "id", "description", "type"]]
    return annot_df.to_dict(orient="records")


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


def df_to_point_layer(
    df,
    name,
    color,
    layer_template=None,
    visible=False,
    xcol="centroid_x",
    ycol="centroid_y",
    zcol="centroid_z",
    idcol="source_row",
    descriptioncol="Cell Type",
):

    annot_dict = df_to_point_annotations(
        df,
        xcol=xcol,
        ycol=ycol,
        zcol=zcol,
        idcol=idcol,
        descriptioncol=descriptioncol,
    )

    return annotations_to_layer(
        layer_template=layer_template,
        annot_dict=annot_dict,
        name=name,
        color=color,
        visible=visible,
    )


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
    # %%
    df = pd.read_csv(
        "data/ribbon_seg_id_runs/run_20260408_110904/ribbon_with_cell_labels.csv"
    )

    # %%
    layer = df_to_point_layer(
        df,
        name="my_point_annotations",
        color="#00c853",
        visible=True,
        xcol="centroid_x",
        ycol="centroid_y",
        zcol="centroid_z",
        idcol="source_row",
        descriptioncol="Cell Type",
    )

    # %%
    base_state = load_state_json("data/state.json")
    state_with_annotations = add_layer_to_state(base_state, layer)
    make_link(state_with_annotations)
    # %%
