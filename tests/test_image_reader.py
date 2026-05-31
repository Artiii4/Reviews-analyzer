from PIL import Image

from src.image_reader import prepare_image


def test_prepare_image_grayscale():
    image = Image.new("RGB", (100, 50), color="white")
    result = prepare_image(image)
    assert result.mode == "L"
