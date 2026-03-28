import neuroglancer

neuroglancer.set_server_bind_address("127.0.0.1", 8080)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    # Original image layer
    s.layers.append(
        name="img",
        layer=neuroglancer.ImageLayer(
            source="precomputed://gs://stroeh_sem_mouse_retina/image/v2"
        ),
    )

    # Segmentation layer 1
    s.layers.append(
        name="stroeh_mouse_retina",
        layer=neuroglancer.SegmentationLayer(
            source="graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina"
        ),
    )

    # Segmentation layer 2
    s.layers.append(
        name="seg",
        layer=neuroglancer.SegmentationLayer(
            source="precomputed://gs://alex_research/stroeh_retina/synapse/250923_ribbon_v2/seg/"
        ),
    )

    s.layers.append(
        name="annotations",
        layer=neuroglancer.AnnotationLayer(
            source="precomputed://http://localhost:9000"
        ),
    )
print(viewer)
