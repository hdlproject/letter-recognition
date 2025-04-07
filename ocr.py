import pdf2image
import pytesseract


class OCR:
    def __init__(self):
        pass

    @staticmethod
    def __pdf_to_img(pdf_file_path):
        return pdf2image.convert_from_path(pdf_file_path)

    @staticmethod
    def __img_to_text(file):
        text = pytesseract.image_to_string(file)
        return text

    def get_text(self, pdf_file_path):
        images = self.__pdf_to_img(pdf_file_path)

        result = ""
        for pg, img in enumerate(images):
            res_per_page = self.__img_to_text(img)
            result = "{} \n {}".format(result, res_per_page)

        f = open("ocr_output.txt", "w")
        f.write(result)
        f.close()

        return result
