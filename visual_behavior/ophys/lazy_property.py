class LazyProperty(object):
    
    def __init__(self, api_method, *args, **kwargs):

        self.api_method = api_method
        self.args = args
        self.kwargs = kwargs
        self.value = None

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if self.value is None:
            self.value = self.calculate()
        return self.value

    def __set__(self, obj, value):
        raise AttributeError("Can't set LazyLoadable attribute")

    def calculate(self):
        return self.api_method(*self.args, **self.kwargs)

class LazyPropertyMixin(object):

    @property
    def LazyProperty(self):
        return LazyProperty
    # LazyProperty = LazyProperty

    def __getattribute__(self, name):

        lazy_class = super(LazyPropertyMixin, self).__getattribute__('LazyProperty')
        curr_attr = super(LazyPropertyMixin, self).__getattribute__(name)
        if isinstance(curr_attr, lazy_class):
            return curr_attr.__get__(curr_attr)
        else:
            return super(LazyPropertyMixin, self).__getattribute__(name)


    def __setattr__(self, name, value):
        if not hasattr(self, name):
            super(LazyPropertyMixin, self).__setattr__(name, value)
        else:
            curr_attr = super(LazyPropertyMixin, self).__getattribute__(name)
            lazy_class = super(LazyPropertyMixin, self).__getattribute__('LazyProperty')
            if isinstance(curr_attr, lazy_class):
                curr_attr.__set__(curr_attr, value)
            else:
                super(LazyPropertyMixin, self).__setattr__(name, value)



def test_lazy_property_mixin():

    import time
    x, y = 3, 4

    class TestApi(object):
        
        def add(self, x, y):
            time.sleep(1)
            return x + y

    class TestClass(LazyPropertyMixin):

        def __init__(self):
            
            self.test_api = TestApi()
            self.x = self.LazyProperty(self.test_api.add, x, y)

    test_class = TestClass()

    t0 = time.time()
    assert test_class.x == x+y
    t1 = time.time()-t0

    t0 = time.time()
    assert test_class.x == x+y
    t2 = time.time()-t0

    assert t1 > 10*t2

    fail = False
    try:
        test_class.x = 'BLARG'
    except AttributeError as e:
        fail = True
    assert fail == True

    class ForbiddenClass(TestClass):
        
        def __init__(self):
            
            self.test_api = TestApi()
            super(ForbiddenClass, self).__init__()
            self.x = self.LazyProperty(self.test_api.add, 3, 4)

    fail = False
    try:
        forbidden_class = ForbiddenClass()
    except AttributeError as e:
        fail = True
    assert fail == True
    

if __name__ == "__main__":

    test_lazy_property_mixin()