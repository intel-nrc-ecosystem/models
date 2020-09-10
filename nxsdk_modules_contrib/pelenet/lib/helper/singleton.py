"""
@desc: This class can be used as a decorator to make any class a singleton
@note: This approach is mainly inspired from:
        - https://stackoverflow.com/a/7346105/2692283
        - https://realpython.com/primer-on-python-decorators/#creating-singletons
"""
class Singleton:
    
    def __init__(self, decoratedClass):
        # Instance does initially not exist
        self._instance = None
        # Store decorated class for its use in 'instance' method
        self.decoratedClass = decoratedClass

    def instance(self, *args, **kwargs):
        # Check if instance already exists
        if self._instance is None:
            # If instance was not already created, create one from the decorated class
            self._instance = self.decoratedClass(*args, **kwargs)
        # In any case: return instance
        return self._instance

    def __call__(self):
        # Raises error if the singelton instance is called directly, like Foo() instead of Foo.instance()
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, instance):
        # Checks if instance is instance of class self._decorated
        return isinstance(instance, self.decoratedClass)
