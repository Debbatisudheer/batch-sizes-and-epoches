Code Breakdown:

    Imports:
        tensorflow as tf: Imports the TensorFlow library for deep learning.
        from tensorflow.keras import layers, models: Imports layers and model building components from Keras, a high-level API on top of TensorFlow.
        import numpy as np: Imports NumPy for numerical computations.
        import matplotlib.pyplot as plt: Imports Matplotlib for plotting training metrics.

    Dummy Data:
        X_train: Creates a random NumPy array representing 1000 training images, each with a shape of (28, 28, 1) (likely grayscale images).
        y_train: Creates a random NumPy array representing labels (categories) for the training images, with 10 possible categories.

    Model Definition:
        model = models.Sequential(): Creates a sequential model, where layers are added one after the other.
        Layers are defined using layers.Conv2D, layers.MaxPooling2D, layers.Flatten, and layers.Dense. These represent convolutional layers, pooling layers, a flattening layer, and fully-connected layers, respectively. The specific configuration defines a Convolutional Neural Network (CNN) architecture.

    Model Compilation:
        model.compile(): Sets up the training process by specifying the optimizer (adam), loss function (sparse_categorical_crossentropy for multi-class classification), and metrics (accuracy).

    Training Loop:
        Defines batch_size and epochs for training.
        Initializes empty lists train_losses and train_accuracies to store training metrics.
        Iterates through epochs:
            Prints the current epoch number.
            Initializes empty lists epoch_losses and epoch_accuracies for the current epoch.
            Iterates through training data in batches using a custom loop:
                Extracts a batch of training images and labels using slicing.
                Trains the model on the batch using model.train_on_batch(), which returns loss and accuracy values.
                Appends the loss and accuracy to the epoch lists.
                Prints batch-level metrics (loss and accuracy).
            Calculates average loss and accuracy for the epoch.
            Appends the averaged metrics to the training lists.
            Plots the training loss and accuracy curves using Matplotlib.

    Model Evaluation (for Demonstration):
        Evaluates the trained model on the same training data (not ideal for real-world scenarios) using model.evaluate(). This gives test loss and accuracy values.


output analysis: 

Epoch 1/5:
Batch 1/31: Loss = 2.323923110961914, Accuracy = 0.03125
Batch 2/31: Loss = 2.3255198001861572, Accuracy = 0.078125
Batch 3/31: Loss = 2.313532590866089, Accuracy = 0.0729166641831398
Batch 4/31: Loss = 2.3291847705841064, Accuracy = 0.0546875
Batch 5/31: Loss = 2.33958101272583, Accuracy = 0.0625
Batch 6/31: Loss = 2.3380203247070312, Accuracy = 0.0572916679084301
Batch 7/31: Loss = 2.3288543224334717, Accuracy = 0.0714285746216774
Batch 8/31: Loss = 2.323298454284668, Accuracy = 0.078125
Batch 9/31: Loss = 2.321098566055298, Accuracy = 0.0763888880610466
Batch 10/31: Loss = 2.3253979682922363, Accuracy = 0.08437500149011612
Batch 11/31: Loss = 2.328965902328491, Accuracy = 0.08238636702299118
Batch 12/31: Loss = 2.3273346424102783, Accuracy = 0.0859375
Batch 13/31: Loss = 2.3245816230773926, Accuracy = 0.09375
Batch 14/31: Loss = 2.3226685523986816, Accuracy = 0.0982142835855484
Batch 15/31: Loss = 2.319406032562256, Accuracy = 0.09791667014360428
Batch 16/31: Loss = 2.3203535079956055, Accuracy = 0.095703125
Batch 17/31: Loss = 2.3184263706207275, Accuracy = 0.0992647036910057
Batch 18/31: Loss = 2.3192481994628906, Accuracy = 0.0972222238779068
Batch 19/31: Loss = 2.3203682899475098, Accuracy = 0.09375
Batch 20/31: Loss = 2.3195300102233887, Accuracy = 0.09687499701976776
Batch 21/31: Loss = 2.3186581134796143, Accuracy = 0.095238097012043
Batch 22/31: Loss = 2.318122386932373, Accuracy = 0.09517045319080353
Batch 23/31: Loss = 2.3187270164489746, Accuracy = 0.09103260934352875
Batch 24/31: Loss = 2.318253517150879, Accuracy = 0.0924479141831398
Batch 25/31: Loss = 2.3176541328430176, Accuracy = 0.09375
Batch 26/31: Loss = 2.316593647003174, Accuracy = 0.09375
Batch 27/31: Loss = 2.3157105445861816, Accuracy = 0.09490741044282913
Batch 28/31: Loss = 2.315762758255005, Accuracy = 0.09375
Batch 29/31: Loss = 2.3154566287994385, Accuracy = 0.09159483015537262
Batch 30/31: Loss = 2.3151862621307373, Accuracy = 0.09166666865348816
Batch 31/31: Loss = 2.3146727085113525, Accuracy = 0.09173387289047241
Batch 32/31: Loss = 2.313138246536255, Accuracy = 0.09399999678134918
Epoch 2/5:
Batch 1/31: Loss = 2.3130133152008057, Accuracy = 0.09302325546741486
Batch 2/31: Loss = 2.3119771480560303, Accuracy = 0.09398496150970459
Batch 3/31: Loss = 2.3119096755981445, Accuracy = 0.09306569397449493
Batch 4/31: Loss = 2.3112595081329346, Accuracy = 0.09574468433856964
Batch 5/31: Loss = 2.310892343521118, Accuracy = 0.09655172377824783
Batch 6/31: Loss = 2.3109657764434814, Accuracy = 0.09731543809175491
Batch 7/31: Loss = 2.3102939128875732, Accuracy = 0.09640523046255112
Batch 8/31: Loss = 2.310178518295288, Accuracy = 0.09633757919073105
Batch 9/31: Loss = 2.309720039367676, Accuracy = 0.09704969078302383
Batch 10/31: Loss = 2.310028314590454, Accuracy = 0.09696969389915466
Batch 11/31: Loss = 2.3100955486297607, Accuracy = 0.09689348936080933
Batch 12/31: Loss = 2.3092782497406006, Accuracy = 0.09971098601818085
Batch 13/31: Loss = 2.3088772296905518, Accuracy = 0.09887005388736725
Batch 14/31: Loss = 2.308814287185669, Accuracy = 0.09944751113653183
Batch 15/31: Loss = 2.30815052986145, Accuracy = 0.10000000149011612
Batch 16/31: Loss = 2.308448553085327, Accuracy = 0.10052909702062607
Batch 17/31: Loss = 2.3081908226013184, Accuracy = 0.09974092990159988
Batch 18/31: Loss = 2.308304786682129, Accuracy = 0.09898477047681808
Batch 19/31: Loss = 2.3084423542022705, Accuracy = 0.09763681888580322
Batch 20/31: Loss = 2.308316469192505, Accuracy = 0.09695121645927429
Batch 21/31: Loss = 2.3082921504974365, Accuracy = 0.09748803824186325
Batch 22/31: Loss = 2.3081791400909424, Accuracy = 0.09741783887147903
Batch 23/31: Loss = 2.3082311153411865, Accuracy = 0.09735023230314255
Batch 24/31: Loss = 2.3078787326812744, Accuracy = 0.0984162911772728
Batch 25/31: Loss = 2.307650089263916, Accuracy = 0.09833333641290665
Batch 26/31: Loss = 2.307305335998535, Accuracy = 0.10043668001890182
Batch 27/31: Loss = 2.307096481323242, Accuracy = 0.10139484703540802
Batch 28/31: Loss = 2.3072454929351807, Accuracy = 0.10126582533121109
Batch 29/31: Loss = 2.3071346282958984, Accuracy = 0.1011410802602768
Batch 30/31: Loss = 2.3069581985473633, Accuracy = 0.10051020234823227
Batch 31/31: Loss = 2.3067872524261475, Accuracy = 0.10090361535549164
Batch 32/31: Loss = 2.3063533306121826, Accuracy = 0.10100000351667404
Epoch 3/5:
Batch 1/31: Loss = 2.3063809871673584, Accuracy = 0.10039369761943817
Batch 2/31: Loss = 2.3060216903686523, Accuracy = 0.09980620443820953
Batch 3/31: Loss = 2.3060059547424316, Accuracy = 0.10114503651857376
Batch 4/31: Loss = 2.3056674003601074, Accuracy = 0.10385338217020035
Batch 5/31: Loss = 2.3055193424224854, Accuracy = 0.10324074327945709
Batch 6/31: Loss = 2.305281639099121, Accuracy = 0.1031021922826767
Batch 7/31: Loss = 2.304957389831543, Accuracy = 0.10296762734651566
Batch 8/31: Loss = 2.304811954498291, Accuracy = 0.10328014194965363
Batch 9/31: Loss = 2.3046562671661377, Accuracy = 0.1035839170217514
Batch 10/31: Loss = 2.3045597076416016, Accuracy = 0.10474137961864471
Batch 11/31: Loss = 2.3046576976776123, Accuracy = 0.1041666641831398
Batch 12/31: Loss = 2.3040578365325928, Accuracy = 0.10570469498634338
Batch 13/31: Loss = 2.3035471439361572, Accuracy = 0.1076158955693245
Batch 14/31: Loss = 2.303356409072876, Accuracy = 0.10743463784456253
Batch 15/31: Loss = 2.302873373031616, Accuracy = 0.10806451737880707
Batch 16/31: Loss = 2.303067207336426, Accuracy = 0.1082802563905716
Batch 17/31: Loss = 2.3027052879333496, Accuracy = 0.10849056392908096
Batch 18/31: Loss = 2.3028903007507324, Accuracy = 0.10791925340890884
Batch 19/31: Loss = 2.303306818008423, Accuracy = 0.10812883079051971
Batch 20/31: Loss = 2.303147554397583, Accuracy = 0.10757575929164886
Batch 21/31: Loss = 2.3029532432556152, Accuracy = 0.10703592747449875
Batch 22/31: Loss = 2.3028206825256348, Accuracy = 0.10724852234125137
Batch 23/31: Loss = 2.303057909011841, Accuracy = 0.10635964572429657
Batch 24/31: Loss = 2.3028829097747803, Accuracy = 0.106575146317482
Batch 25/31: Loss = 2.3027071952819824, Accuracy = 0.1071428582072258
Batch 26/31: Loss = 2.3023781776428223, Accuracy = 0.10699152201414108
Batch 27/31: Loss = 2.302205801010132, Accuracy = 0.10824022442102432
Batch 28/31: Loss = 2.3023416996002197, Accuracy = 0.10738950222730637
Batch 29/31: Loss = 2.302203893661499, Accuracy = 0.1075819656252861
Batch 30/31: Loss = 2.3020148277282715, Accuracy = 0.10844594240188599
Batch 31/31: Loss = 2.3018875122070312, Accuracy = 0.1092914417386055
Batch 32/31: Loss = 2.3011996746063232, Accuracy = 0.10999999940395355
Epoch 4/5:
Batch 1/31: Loss = 2.30137300491333, Accuracy = 0.11048812419176102
Batch 2/31: Loss = 2.3009443283081055, Accuracy = 0.11129242926836014
Batch 3/31: Loss = 2.3009731769561768, Accuracy = 0.11175710707902908
Batch 4/31: Loss = 2.3007006645202637, Accuracy = 0.11317135393619537
Batch 5/31: Loss = 2.3005149364471436, Accuracy = 0.11329113692045212
Batch 6/31: Loss = 2.300436496734619, Accuracy = 0.11340852081775665
Batch 7/31: Loss = 2.2999789714813232, Accuracy = 0.11352357268333435
Batch 8/31: Loss = 2.299863815307617, Accuracy = 0.11425061523914337
Batch 9/31: Loss = 2.2997055053710938, Accuracy = 0.1146593689918518
Batch 10/31: Loss = 2.2998359203338623, Accuracy = 0.11536144465208054
Batch 11/31: Loss = 2.2999022006988525, Accuracy = 0.11485680192708969
Batch 12/31: Loss = 2.2994344234466553, Accuracy = 0.11495272070169449
Batch 13/31: Loss = 2.299069881439209, Accuracy = 0.1156323179602623
Batch 14/31: Loss = 2.2989182472229004, Accuracy = 0.11600928008556366
Batch 15/31: Loss = 2.298527956008911, Accuracy = 0.11724138259887695
Batch 16/31: Loss = 2.298524856567383, Accuracy = 0.11702733486890793
Batch 17/31: Loss = 2.2982337474823, Accuracy = 0.1173814907670021
Batch 18/31: Loss = 2.2982726097106934, Accuracy = 0.11661073565483093
Batch 19/31: Loss = 2.298214912414551, Accuracy = 0.11723946779966354
Batch 20/31: Loss = 2.2980775833129883, Accuracy = 0.11730769276618958
Batch 21/31: Loss = 2.2976973056793213, Accuracy = 0.1173747256398201
Batch 22/31: Loss = 2.2975618839263916, Accuracy = 0.11798056215047836
Batch 23/31: Loss = 2.2978317737579346, Accuracy = 0.11750535666942596
Batch 24/31: Loss = 2.2976038455963135, Accuracy = 0.11703822016716003
Batch 25/31: Loss = 2.297314167022705, Accuracy = 0.1173684224486351
Batch 26/31: Loss = 2.296977996826172, Accuracy = 0.11769311130046844
Batch 27/31: Loss = 2.2967171669006348, Accuracy = 0.11827122420072556
Batch 28/31: Loss = 2.2966561317443848, Accuracy = 0.11806981265544891
Batch 29/31: Loss = 2.2965097427368164, Accuracy = 0.11787168681621552
Batch 30/31: Loss = 2.296567678451538, Accuracy = 0.11792929470539093
Batch 31/31: Loss = 2.2964980602264404, Accuracy = 0.11848697066307068
Batch 32/31: Loss = 2.2953922748565674, Accuracy = 0.11949999630451202
Epoch 5/5:
Batch 1/31: Loss = 2.2957520484924316, Accuracy = 0.119295634329319
Batch 2/31: Loss = 2.2953507900238037, Accuracy = 0.11934055387973785
Batch 3/31: Loss = 2.295269012451172, Accuracy = 0.119140625
Batch 4/31: Loss = 2.2949278354644775, Accuracy = 0.12039728462696075
Batch 5/31: Loss = 2.294752359390259, Accuracy = 0.12043268978595734
Batch 6/31: Loss = 2.29461669921875, Accuracy = 0.12046755850315094
Batch 7/31: Loss = 2.294038772583008, Accuracy = 0.12073863297700882
Batch 8/31: Loss = 2.293837308883667, Accuracy = 0.12124060094356537
Batch 9/31: Loss = 2.2935802936553955, Accuracy = 0.12126865983009338
Batch 10/31: Loss = 2.2934257984161377, Accuracy = 0.12175925821065903
Batch 11/31: Loss = 2.293489933013916, Accuracy = 0.12132352590560913
Batch 12/31: Loss = 2.292670488357544, Accuracy = 0.12249087542295456
Batch 13/31: Loss = 2.2919466495513916, Accuracy = 0.12432064861059189
Batch 14/31: Loss = 2.291635513305664, Accuracy = 0.12477517873048782
Batch 15/31: Loss = 2.291287422180176, Accuracy = 0.12522321939468384
Batch 16/31: Loss = 2.2911131381988525, Accuracy = 0.1252216249704361
Batch 17/31: Loss = 2.290843963623047, Accuracy = 0.1261003464460373
Batch 18/31: Loss = 2.29097318649292, Accuracy = 0.1258741319179535
Batch 19/31: Loss = 2.290741205215454, Accuracy = 0.1260850727558136
Batch 20/31: Loss = 2.2906227111816406, Accuracy = 0.12672413885593414
Batch 21/31: Loss = 2.2901642322540283, Accuracy = 0.12756849825382233
Batch 22/31: Loss = 2.2899224758148193, Accuracy = 0.12818877398967743
Batch 23/31: Loss = 2.289907455444336, Accuracy = 0.12858952581882477
Batch 24/31: Loss = 2.2895901203155518, Accuracy = 0.12940436601638794
Batch 25/31: Loss = 2.2892961502075195, Accuracy = 0.1302083283662796
Batch 26/31: Loss = 2.288667678833008, Accuracy = 0.1312086135149002
Batch 27/31: Loss = 2.2882282733917236, Accuracy = 0.13116776943206787
Batch 28/31: Loss = 2.2886106967926025, Accuracy = 0.1313316971063614
Batch 29/31: Loss = 2.2885260581970215, Accuracy = 0.13108766078948975
Batch 30/31: Loss = 2.288562536239624, Accuracy = 0.13104838132858276
Batch 31/31: Loss = 2.288515567779541, Accuracy = 0.13120993971824646
Batch 32/31: Loss = 2.2868409156799316, Accuracy = 0.13199999928474426
32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.1966 - loss: 2.2393
Test Loss: 2.219108819961548, Test Accuracy: 0.2160000056028366
