from utils import *

class CNN_Spooffer():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, image, target_class, minimum_confidence, files, ):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.minimum_confidence = minimum_confidence
        self.files = files
        # Generate a random image
        self.image = image
        # Create the folder to export images if not exists
        cv2.imwrite(self.files['original_image_path'], recreate_image(self.image))

    def gradAscent(self):
        # Define optimizer for the image
        optimizer = SGD([self.image], lr=0.7)
        for i in range(1, 500):
            # Forward
            output = self.model(self.image)
            # Get confidence from softmax
            target_confidence = functional.softmax(output,dim=1)[0][self.target_class].data.item()
            if target_confidence > self.minimum_confidence:
                break
            # Target specific class
            class_loss = -output[0, self.target_class]
            print(f'\tIteration: {i} Target confidence', "{0:.4f}".format(target_confidence))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
        spooffed_image = self.image
        # Recreate image
        self.image = recreate_image(self.image)
        # Save image
        cv2.imwrite(self.files['spooffed_image_path'], self.image)
        return spooffed_image
