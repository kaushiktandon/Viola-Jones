class FeatureRepresentation(object):
    def __init__(self, position, width, height, threshold, polarity):
        self.position = position
        self.tl = position
        self.br = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity

    def GetClassification(self, intImage):
        score = self.ComputeScore(intImage)
        return 1 if score < self.polarity * self.threshold else -1


class FeatureTwoVertical(FeatureRepresentation):
    dimen = (1, 2)

    def ComputeScore(self, intImage):
        first = intImage.GetAreaSum(
            self.tl, (self.tl[0] +
                      self.width, self.tl[1] + self.height / 2))

        second = intImage.GetAreaSum(
            (self.tl[0], self.tl[1] +
             self.height / 2), self.br)

        score = first - second
        return score


class FeatureTwoHorizontal(FeatureRepresentation):
    dimen = (2, 1)

    def ComputeScore(self, intImage):
        first = intImage.GetAreaSum(self.tl, (self.tl[0] + self.width / 2, self.tl[1] + self.height))
        second = intImage.GetAreaSum((self.tl[0] + self.width / 2, self.tl[1]), self.br)
        score = first - second
        return score


class FeatureThreeHorizontal(FeatureRepresentation):
    dimen = (3, 1)

    def ComputeScore(self, intImage):
        first = intImage.GetAreaSum(self.tl, (self.tl[0] + self.width / 3, self.tl[1] + self.height))
        second = intImage.GetAreaSum((self.tl[0] + self.width / 3, self.tl[1]), (self.tl[0] + 2 * self.width / 3, self.tl[1] + self.height))
        third = intImage.GetAreaSum((self.tl[0] + 2 * self.width / 3, self.tl[1]), self.br)
        score = first - second + third
        return score


class FeatureThreeVertical(FeatureRepresentation):
    dimen = (1, 3)

    def ComputeScore(self, intImage):
        first = intImage.GetAreaSum(self.tl, (self.br[0], self.tl[1] + self.height / 3))
        second = intImage.GetAreaSum((self.tl[0], self.tl[1] + self.height / 3), (self.br[0], self.tl[1] + 2 * self.height / 3))
        third = intImage.GetAreaSum((self.tl[0], self.tl[1] + 2 * self.height / 3), self.br)
        score = first - second + third
        return score


class FeatureFourHorizontal(FeatureRepresentation):
    dimen = (2, 2)

    def ComputeScore(self, intImage):
        first = intImage.GetAreaSum(self.tl, (self.tl[0] + self.width / 2, self.tl[1] + self.height / 2))
        # top right area
        second = intImage.GetAreaSum((self.tl[0] + self.width / 2, self.tl[1]), (self.br[0], self.tl[1] + self.height / 2))
        # bottom left area
        third = intImage.GetAreaSum((self.tl[0], self.tl[1] + self.height / 2), (self.tl[0] + self.width / 2, self.br[1]))
        # bottom right area
        fourth = intImage.GetAreaSum((self.tl[0] + self.width / 2, self.tl[1] + self.height / 2), self.br)
        score = first - second - third + fourth
        return score


class FEATURE_TYPES:
    TYPE_TWO_VERTICAL = FeatureTwoVertical
    TYPE_TWO_HORIZONTAL = FeatureTwoHorizontal
    TYPE_THREE_HORIZONTAL = FeatureThreeHorizontal
    TYPE_THREE_VERTICAL = FeatureThreeVertical
    TYPE_FOUR_HORIZONTAL = FeatureFourHorizontal

    ALL = [TYPE_TWO_VERTICAL, TYPE_TWO_HORIZONTAL, TYPE_THREE_HORIZONTAL,
           TYPE_THREE_VERTICAL, TYPE_FOUR_HORIZONTAL]