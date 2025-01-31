




def split_data(data, type):
    for i in range(len(data["image"])):
        data["image"][i].save(fr"breastcancer-ultrasound-images\data\{type}\images\{i}.png")
        with open(fr"breastcancer-ultrasound-images\data\{type}\labels\{i}.txt", "w+") as f:
            f.write(f"{data["label"][i]}")
    