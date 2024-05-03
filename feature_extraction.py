import numpy as np
def OCR_raw_data(imagefile, labelfile):

    with open(imagefile, 'rb') as file:
        lines = file.readlines()

    # Initialize variables
    image_data = []
    images = []

    # Parse the lines and populate the list of images
    for i, line in enumerate(lines):
        line = line.decode()  # Convert bytes to string and remove newline character
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        
        # Check for non-empty line
        if line:
            row_data = [1 if c == '#' else 0 for c in line]  # Convert symbols to pixel values
            image_data.extend(row_data)
            
            # Check for end of image (28 lines)
            if len(image_data) == 28*28:
                images.append(image_data)
                image_data = []  # Reset image_data
        else:
            # Check for start of new image (empty line followed by line with symbols)
            if i < len(lines) - 1 and lines[i + 1].decode():  # Check if next line is not empty
                image_data = []  # Reset image_data
    # Convert the list of images to a matrix
    X = np.array(images)
    #print(X.shape)  # Should print (number_of_images, 16)

    with open(labelfile, 'rb') as file:
        lineLab = file.readlines()
    labels = []
    for i, line in enumerate(lineLab):
        line = line.decode()
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        labels.append(int(line))
    Y = labels
    #print(len(Y))
    return X, Y
#OCR_raw_data('digitdata/trainingimages', 'digitdata/traininglabels')

def OCR_feature_data(imagefile, labelfile, width, length):
    
    with open(imagefile, 'rb') as file:
        lines = file.readlines()

    # Initialize variables
    image_data = []
    images = []

    # Parse the lines and populate the list of images
    for i, line in enumerate(lines):
        line = line.decode()  # Convert bytes to string and remove newline character
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        
        # Check for non-empty line
        if line:
            row_data = [1 if c == '#' else 0 for c in line]  # Convert symbols to pixel values
            image_data.append(row_data)
            
            # Check for end of image (28 lines)
            if len(image_data) == 28:
                # Convert the list of lists (28x28) to a numpy array
                image_array = np.array(image_data)
                
                # Extract features by counting 0s in each 7x7 square
                features = []
                for k in range(0, 28, length):
                    for j in range(0, 28, width):
                        square = image_array[k:k+length, j:j+width]
                        count_zeros = np.sum(square == 1)
                        features.append(count_zeros)
                
                # Flatten the 4x4 feature matrix
                features = np.array(features).flatten()
                
                images.append(features)
                image_data = []  # Reset image_data
        else:
            # Check for start of new image (empty line followed by line with symbols)
            if i < len(lines) - 1 and lines[i + 1].decode():  # Check if next line is not empty
                image_data = []  # Reset image_data
    # Convert the list of images to a matrix
    X = np.array(images)
    #print(X.shape)  # Should print (number_of_images, 16)

    with open(labelfile, 'rb') as file:
        lineLab = file.readlines()
    labels = []
    for i, line in enumerate(lineLab):
        line = line.decode()
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        labels.append(int(line))
    Y = labels
    #print(len(Y))
    return X, Y
#OCR_feature_data('digitdata/trainingimages', 'digitdata/traininglabels', 7, 7)


def face_raw_data(imagefile, labelfile):
    with open(imagefile, 'rb') as file:
        lines = file.readlines()
    # Initialize variables
    image_data = []
    images = []

    # Parse the lines and populate the list of images
    for i, line in enumerate(lines):
        line = line.decode()  # Convert bytes to string and remove newline character
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        row_data = [1 if c == '#' else 0 for c in line]  # Convert symbols to pixel values

        image_data.extend(row_data)
        # Check for end of image (70 lines)
        if (i+1) % 70==0:
            images.append(image_data)
            image_data = []
    # Convert the list of images to a matrix
    X = np.array(images)

    with open(labelfile, 'rb') as file:
        lineLab = file.readlines()
    labels = []
    for i, line in enumerate(lineLab):
        line = line.decode()
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        labels.append(int(line))
    Y = labels

    return X, Y
        
#face_raw_data("facedata/facedatatrain", "facedata/facedatatrainlabels")


def face_feature_data(imagefile, labelfile,length, width):
    with open(imagefile, 'rb') as file:
        lines = file.readlines()
    # Initialize variables
    image_data = []
    images = []

    # Parse the lines and populate the list of images
    for i, line in enumerate(lines):
        line = line.decode()  # Convert bytes to string and remove newline character
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        row_data = [1 if c == '#' else 0 for c in line]  # Convert symbols to pixel values

        image_data.append(row_data)
        # Check for end of image (70 lines)
        if (i+1) % 70==0:
        # Convert the list of lists (28x28) to a numpy array
            image_array = np.array(image_data)
            
            # Extract features by counting 0s in each 7x7 square
            features = []
            for k in range(0, 70, length):
                for j in range(0, 60, width):
                    square = image_array[k:k+length, j:j+width]
                    count_zeros = np.sum(square == 1)
                    features.append(count_zeros)
            
            # Flatten the 4x4 feature matrix
            features = np.array(features).flatten()
            
            images.append(features)
            image_data = []  # Reset image_data
            
    # Convert the list of images to a matrix
    X = np.array(images)
    #print(X.shape)
    with open(labelfile, 'rb') as file:
        lineLab = file.readlines()
    labels = []
    for i, line in enumerate(lineLab):
        line = line.decode()
        lineSplit= line.split("\n")
        line = "".join(lineSplit)
        labels.append(int(line))
    Y = labels
    #print(len(Y))
    return X, Y
        
#face_feature_data("facedata/facedatatrain", "facedata/facedatatrainlabels", 10, 10)
