from video import VideoReader
from cnn import LetterRecognitionModel
from cnn_hyperopt import LetterRecognitionModelHyperOpt
from ocr import OCR
from nlp import NLP

# # hyperparameter optimization step
# if __name__ == '__main__':
#     lrm = LetterRecognitionModelHyperOpt()
#     lrm.get_data()
#     # best = lrm.find_best_model(100)
#     best = lrm.find_best_model_with_persistence(102, 3)
#
#     print("Best: {}".format(best))

# # model testing step
# if __name__ == '__main__':
#     lrm = LetterRecognitionModel()
#     lrm.get_data()
#     lrm.train()
#
#     print(lrm.predict(lrm.X_test[0]))

# # real program
# if __name__ == '__main__':
#     vr = VideoReader()
#     lrm = LetterRecognitionModel()
#     lrm.get_data()
#     lrm.train()
#
#     count = 0
#     for (ret, frame) in vr.read():
#         count += 1
#         if count == 10:
#             print(lrm.predict(frame))
#             count = 0
#
#         if vr.exit_condition():
#             break
#
#     vr.exit()

# find entities from pdf
if __name__ == '__main__':
    # convert pdf to text using Google Tesseract
    ocr = OCR()
    result = ocr.get_text('ocr_input.pdf')
    print(result)

    # extract entities using ChatGPT
    # nlp = NLP()
    # nlp.get_ner(result)
