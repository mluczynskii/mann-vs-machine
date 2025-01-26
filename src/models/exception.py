
class NotTrainedError(Exception):
    pass

not_fitted_message = ("The vectorizer has not been fitted yet. "
                      "You must preprocess a training set first.")

class NotFittedError(Exception):
    
    def __init__(self, msg=not_fitted_message, *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
