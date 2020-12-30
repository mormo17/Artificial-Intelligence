import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) < 0:
            return -1

        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        batch_size = 1
        accuracy_achieved = False
        while not accuracy_achieved:
            accuracy_achieved = True
            for input, res in dataset.iterate_once(batch_size):
                prediction = self.get_prediction(input)
                # print(prediction)
                if prediction != nn.as_scalar(res):
                    accuracy_achieved = False
                    self.w.update(input, nn.as_scalar(res))

            if accuracy_achieved:
                break
            




class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.first_hidden_layer = nn.Parameter(1, 40)
        self.second_hidden_layer = nn.Parameter(40, 150)
        self.third_hidden_layer = nn.Parameter(150, 1)

        self.first_hidden_layer_bias = nn.Parameter(1, 40)
        self.second_hidden_layer_bias = nn.Parameter(1, 150)
        self.third_hidden_layer_bias = nn.Parameter(1, 1)

        self.batch_size = 200
        self.learning_rate = -0.01

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        first_res = nn.Linear(x, self.first_hidden_layer)
        first_res = nn.AddBias(first_res, self.first_hidden_layer_bias)
        first_res = nn.ReLU(first_res)

        second_res = nn.Linear(first_res, self.second_hidden_layer)
        second_res = nn.AddBias(second_res, self.second_hidden_layer_bias)
        second_res = nn.ReLU(second_res)

        output = nn.Linear(second_res, self.third_hidden_layer)
        output = nn.AddBias(output, self.third_hidden_layer_bias)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """

        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """

        accuracy_achieved = False
        while not accuracy_achieved:
            for input, res in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(input, res)
                parameters = [self.first_hidden_layer, self.first_hidden_layer_bias,
                                  self.second_hidden_layer, self.second_hidden_layer_bias,
                                  self.third_hidden_layer, self.third_hidden_layer_bias]
                gradients_output = nn.gradients(loss, parameters)

                self.first_hidden_layer.update(gradients_output[0], self.learning_rate)
                self.first_hidden_layer_bias.update(gradients_output[1], self.learning_rate)
                self.second_hidden_layer.update(gradients_output[2], self.learning_rate)
                self.second_hidden_layer_bias.update(gradients_output[3], self.learning_rate)
                self.third_hidden_layer.update(gradients_output[4], self.learning_rate)
                self.third_hidden_layer_bias.update(gradients_output[5], self.learning_rate)

                if nn.as_scalar(loss) < 0.02:
                    accuracy_achieved = True

            if accuracy_achieved:
                break

        

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.first_hidden_layer = nn.Parameter(784, 512)
        self.second_hidden_layer = nn.Parameter(512, 256)
        self.third_hidden_layer = nn.Parameter(256, 128)
        self.fourth_hidden_layer = nn.Parameter(128, 64)
        self.fifth_hidden_layer = nn.Parameter(64, 10)

        self.first_hidden_layer_bias = nn.Parameter(1, 512)
        self.second_hidden_layer_bias = nn.Parameter(1, 256)
        self.third_hidden_layer_bias = nn.Parameter(1, 128)
        self.fourth_hidden_layer_bias = nn.Parameter(1, 64)
        self.fifth_hidden_layer_bias = nn.Parameter(1,10)

        self.batch_size = 200
        self.learning_rate = -0.95

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        first_res = nn.Linear(x, self.first_hidden_layer)
        first_res = nn.AddBias(first_res, self.first_hidden_layer_bias)
        first_res = nn.ReLU(first_res)

        second_res = nn.Linear(first_res, self.second_hidden_layer)
        second_res = nn.AddBias(second_res, self.second_hidden_layer_bias)
        second_res = nn.ReLU(second_res)

        third_res = nn.Linear(second_res, self.third_hidden_layer)
        third_res = nn.AddBias(third_res, self.third_hidden_layer_bias)
        third_res = nn.ReLU(third_res)

        fourth_res = nn.Linear(third_res, self.fourth_hidden_layer)
        fourth_res = nn.AddBias(fourth_res, self.fourth_hidden_layer_bias)
        fourth_res = nn.ReLU(fourth_res)

        output = nn.Linear(fourth_res, self.fifth_hidden_layer)
        output = nn.AddBias(output, self.fifth_hidden_layer_bias)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        
        accuracy_achieved = False
        while not accuracy_achieved:
            for input, res in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(input, res)
                parameters = [self.first_hidden_layer, self.first_hidden_layer_bias,
                                  self.second_hidden_layer, self.second_hidden_layer_bias,
                                  self.third_hidden_layer, self.third_hidden_layer_bias,
                                  self.fourth_hidden_layer, self.fourth_hidden_layer_bias,
                                  self.fifth_hidden_layer, self.fifth_hidden_layer_bias]
                gradients_output = nn.gradients(loss, parameters)

                self.first_hidden_layer.update(gradients_output[0], self.learning_rate)
                self.first_hidden_layer_bias.update(gradients_output[1], self.learning_rate)
                self.second_hidden_layer.update(gradients_output[2], self.learning_rate)
                self.second_hidden_layer_bias.update(gradients_output[3], self.learning_rate)
                self.third_hidden_layer.update(gradients_output[4], self.learning_rate)
                self.third_hidden_layer_bias.update(gradients_output[5], self.learning_rate)
                self.fourth_hidden_layer.update(gradients_output[6], self.learning_rate)
                self.fourth_hidden_layer_bias.update(gradients_output[7], self.learning_rate)
                self.fifth_hidden_layer.update(gradients_output[8], self.learning_rate)
                self.fifth_hidden_layer_bias.update(gradients_output[9], self.learning_rate)

                if dataset.get_validation_accuracy() >= 0.975:
                    accuracy_achieved = True

            if accuracy_achieved:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.batch_size = 20
        self.hidden_weight = 200
        self.learning_rate = -0.1
        
        self.first_hidden_layer_weight = nn.Parameter(self.num_chars, self.hidden_weight)
        self.second_hidden_layer_weight = nn.Parameter(self.hidden_weight, self.hidden_weight)
        self.output_weight = nn.Parameter(self.hidden_weight, len(self.languages))


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """

        result = xs[0]       # Dimensions: batch_size x num_chars
        result = nn.ReLU(nn.Linear(result, self.first_hidden_layer_weight))    # Dimensions: batch_size x hidden_weight

        for x in xs[1:]:
            x_i = nn.Linear(x, self.first_hidden_layer_weight)
            result = nn.ReLU(nn.Add(x_i, nn.Linear(result, self.second_hidden_layer_weight)))

        output = nn.Linear(result, self.output_weight)

        return output


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        accuracy_achieved = False
        while not accuracy_achieved:
            for input, res in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(input, res)
                parameters = [self.first_hidden_layer_weight, self.second_hidden_layer_weight, self.output_weight]
                gradients_output = nn.gradients(loss, parameters)

                self.first_hidden_layer_weight.update(gradients_output[0], self.learning_rate)
                self.second_hidden_layer_weight.update(gradients_output[1], self.learning_rate)
                self.output_weight.update(gradients_output[2], self.learning_rate)

                if dataset.get_validation_accuracy() >= 0.89:
                    accuracy_achieved = True
                elif dataset.get_validation_accuracy() >= 0.85:
                    self.learning_rate = -0.02
                elif dataset.get_validation_accuracy() >= 0.83:
                    self.learning_rate = -0.03
                elif dataset.get_validation_accuracy() >= 0.8:
                    self.learning_rate = -0.04
                elif dataset.get_validation_accuracy() >= 0.75:
                    self.learning_rate = -0.05
                elif dataset.get_validation_accuracy() >= 0.7:
                    self.learning_rate = -0.06
                else:
                    self.learning_rate = -0.1


            if accuracy_achieved:
                break
