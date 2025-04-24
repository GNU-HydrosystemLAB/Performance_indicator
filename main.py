from util import Performance, generate_masks_in_directory
import matplotlib.pyplot as plt


for dataset in [["CIP",False],["CP",False],["PP",False],["PVC",True]]:
    #Performance_Evaluation
    P = Performance("weights/PPResNet101.pt",
                    "E:/code/Prediction/"+dataset[0],
                    water = dataset[1], pipe_type = dataset[0])

    P.Perfomance_eval()
    result = P.MAPE()

    print(f"dataset : {dataset[0]}\ndataset num : {len(P.mapping_result)}\nmapping MAPE : {round(100-result,3)}% accuracy ")




    for img in P.mapping_map:
        plt.imshow(img)
        plt.show()