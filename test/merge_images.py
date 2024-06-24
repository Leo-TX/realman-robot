def merge_images(row, column, images_path, save_path=None):
    images = []
    for i in range(0, row * column):
        image_path = os.path.join(images_path, f"trajectory_{i}.png")
        image = Image.open(image_path)
        images.append(image)
    width, height = images[0].size
    result_width = width * column
    result_height = height * row
    result_image = Image.new('RGB', (result_width, result_height))
    for i in range(row * column):
        x = (i % column) * width
        y = (i // column) * height
        result_image.paste(images[i], (x, y))
    if save_path:
        result_image.save(save_path)