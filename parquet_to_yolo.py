




def split_data(data, type):
    for i in range(len(data["image"])):
        cancer_type = data["label"][i]
        if cancer_type == 0: 
            data["image"][i].save(fr"C:\Users\metal\Desktop\ExamensArbete\datasets\dataset\{type}\benign\{i}.png")
        if cancer_type == 1:
            data["image"][i].save(fr"C:\Users\metal\Desktop\ExamensArbete\datasets\dataset\{type}\malignant\{i}.png")
        if cancer_type == 2:
            data["image"][i].save(fr"C:\Users\metal\Desktop\ExamensArbete\datasets\dataset\{type}\normal\{i}.png")
                