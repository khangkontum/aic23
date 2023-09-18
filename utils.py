import re
import numpy as np


def get_key_frame(model, video, keyframe):
    for sample in model.dataset:
        if sample["video"] == video and sample["frameid"] == keyframe:
            return sample["mapped_frameid"], sample["youtube_url"]
    
    return "undifined", "undifined"

def extract_query(text: str):
    matches = re.findall("([\w\s]+)\s+\$(\d+)", text)
    x = [x[0] for x in matches]
    y = np.array(
        [
            float(y[1] if y[1] != "0" else "0.1")
            for y in matches
        ]
    )
    y = y / y.sum()

    return [(xx, yy) for xx, yy in zip(x, y)]


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


if __name__ == "__main__":
    test_string = "first sentence $10 second sentence $0"
    print(extract_query(test_string))

    a = np.random.rand(3, 1)

    print(a)
    print(shift(a, -1, 0.0))
    print(a + a)
