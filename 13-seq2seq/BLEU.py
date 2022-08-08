def modified_bleu(predicted, labels, n):
    #predicted = [0, 20, 35, ...]
    # labels = [
    #     label1,
    #     label2,
    #     ...
    # ]
    #labels = [label] 
    #label = [30, 25, ...]
    predicted_dict = {}
    predicted_len = len(predicted)
    if n > predicted_len:return 0.0
    for i in range(predicted_len - n + 1):
        t = predicted[i:i+n]
        t = tuple(t)
        predicted_dict[t] = predicted_dict.get(t, 0) + 1
    labels_dict = []
    for label in labels:
        label_dict = {}
        for i in range(len(label) - n + 1):
            t = label[i:i+n]
            t = tuple(t)
            label_dict[t] = label_dict.get(t, 0) + 1
        labels_dict.append(label_dict)


    min_sum = 0
    predicted_sum = 0
    for word, times in predicted_dict.items():
        # label_times = []
        # Max = 0
        # for label_dict in labels_dict:
        #     label_times.append(label_dict.get(word, 0))
        #     # Max = max(Max, label_dict.get(word, 0))
        # Max = max(label_times)
        Max = max(label_dict.get(word, 0) for label_dict in labels_dict)
        Min = min(Max, times)
        predicted_sum += times
        min_sum += Min
    return min_sum/predicted_sum

    