import cv2
import pytesseract
import re

# Should be an environment variable, or argument... But yeah... this works for now. Setup accordingly!
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load the file into openCv, check res folder for more examples
file = cv2.imread('res/marketItemSmall6.png')

# We want to work in RGB, since most pics will have a dark background, grayscale worsens our confiability
rgb = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

# Usually, our screenshots will be to small for good confiability, but 2x is too much, 1.5 seems to be enough - could be refined
resized = cv2.resize(rgb, None, fx=1.5, fy=1.5)

# Resized image leaves us with too blurred image, and since we are working with numbers, sharper = better!
# Using Digital unsharp masking technique, pretty simple yet very effective, read more on wikipedia
# Ref: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
blurred = cv2.GaussianBlur(resized, (0,0), 2)
mask = cv2.subtract(resized, blurred)
img = cv2.add(resized, mask)


hImg, wImg, _ = img.shape

# Pytesseract setup

# ! languages -> French + Portuguese + English, have your trainedData files ready or remove the -l argument
config = r'--oem 3 --psm 6 -l fra+por+eng'
output = pytesseract.Output.DICT # * .DATAFRAME for pandas lib
boxes = pytesseract.image_to_data(img, config=config, output_type=output)

# So, we are looking for a number over 1.000, and for sure it has a dot or comma recognized, could be 1.000.000, but not 100 neither .100.
# But 0.100 do pass tho.
number_pattern = '.+(\.|\,)\d{3}'

# Just to clarify: Text -> Actual data read, Conf -> Confiability variable, the more, the better
for i in range(len(boxes['text'])):
    # Ignore unreliable texts!
    if boxes['conf'][i] < 80: continue

    #Pytesseract also gives us Coordinates!
    (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])

    # Using our regex to filter OCR data
    if(re.match(number_pattern, boxes['text'][i])):        
        print(f"!! MATCH {boxes['text'][i]} !!")
        cv2.rectangle(img, (x, y), (w+x,h+y), (0, 0, 255), 1)
        cv2.putText(img, boxes['text'][i], (x, y+35), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 1)

    print(f"Confiability: {boxes['conf'][i]} -> Text: {boxes['text'][i]}") # Easy debug, just to keep track, remove as needed
#if end

# Print our image where the OCR found the data and we parsed :)
cv2.imshow('Result', img)
cv2.waitKey(0)
