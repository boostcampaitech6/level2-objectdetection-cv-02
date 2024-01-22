# How to use WBF Ensemble

1. In wbf_ensemble.ipynb file, you can change the path of the submission files and the weight of each model.

2. you can change the weight(confidence) of each model in the following code.

   ```
    csv_data = [csv_data0, csv_data2, csv_data3, csv_data4]

    weights = [1, 1, 2.5, 1]
   ```

3. you can change the threshold of each model in the following code.

   ```
    iou_thr = 0.5
    skip_box_thr = 0.0001
   ```

4. run wbf_ensemble.ipynb, you can get the submission_ensemble.csv file.