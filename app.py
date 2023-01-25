import time

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

generator = tf.keras.models.load_model("models/gen_SRGAN_10k_e70/")


def translate_image(image):
    return (
        np.clip(
            generator(image[np.newaxis, :] / 127.5 - 1, training=False).numpy()[
                0, :, :, ::-1
            ]
            + 1,
            0,
            2,
        )
        / 2
    )


def app():
    st.set_page_config(page_title="Super-resolution")
    st.title("Super-resolution for face images")
    file = st.file_uploader(
        "Upload your own image of a face! We will scale it to 128x128 and then upscale it to 256x256 with our GAN!",
        type=["jpg"],
    )

    if file is None:
        image = cv2.imread("data/sample_img_128.jpg")
    else:
        image = cv2.resize(cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1), dsize=(128, 128))

    start = time.process_time()
    translated_image = translate_image(image)
    inference_time = time.process_time() - start

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.header("Original")
        st.image(image[..., ::-1], use_column_width=True, width=256)
    with col2:
        st.header("SRGAN upscaled")
        st.image(translated_image, use_column_width=True, clamp=True, width=256)

    st.write(
        f"Upscaling the image took {inference_time:.3f} seconds! \n\n Code available at [GitHub](https://github.com/mserek/face_superresolution)."
    )


if __name__ == "__main__":
    app()
