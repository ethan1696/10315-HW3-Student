from .test_lib import Test, TestSuite
from src import add, Sigmoid, Linear, SoftmaxLoss, NeuralNetwork_2HL, initialize_random_array, train_NN
import numpy as np
import numdifftools as nd

ABSOLUTE_TOLERANCE = 1e-5

################################################### Testing add ###################################################

add_tests = TestSuite("Testing add()")

def add_test_1():
    assert add(1, 2) == 3
add_tests.add_test(Test("Testing add(1, 2) == 3", add_test_1))

def add_test_2():
    assert add(2, 3) == 5
add_tests.add_test(Test("Testing add(2, 3) == 5", add_test_2))

def add_test_3():
    assert add(10, -5) == 5
add_tests.add_test(Test("Testing add(10, -5) == 5", add_test_3))

################################################# Testing Sigmoid #################################################

sigmoid_tests = TestSuite("Testing Sigmoid Layer")

def sigmoid_forward_test_1():
    sigmoid_layer = Sigmoid()

    input = np.array([[0.5326409463227603]])

    expected_output = np.array([[0.6300988598509971]])

    actual_output = sigmoid_layer.forward(input)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

sigmoid_tests.add_test(Test("Testing Sigmoid forward pass on 1x1 array", sigmoid_forward_test_1))

def sigmoid_forward_test_2():
    sigmoid_layer = Sigmoid()

    input = np.array([
        [0.3110270196095799, 0.2664414768620439, 0.07351249143337857, 0.41081978656563556, 0.8522708014843169], 
        [0.7280878197559898, 0.9981729109671269, 0.9153659197236053, 0.7466358675460999, 0.38958331695272674], 
        [0.5944529331890037, 0.7795757592818462, 0.9593143271025757, 0.045265002300454715, 0.43337187582871517], 
        [0.886890520798062, 0.9764632643975467, 0.5543955044974475, 0.08316446258450394, 0.7656175433301471]
    ])

    expected_output = np.array([
        [0.5771359251537383, 0.5662190850191918, 0.5183698509149008, 0.6012844319887661, 0.7010432778485027], 
        [0.6743855176108632, 0.7306991995058799, 0.7140969397153695, 0.6784452301259657, 0.5961823873130648], 
        [0.6443862018151888, 0.6855886731486786, 0.7229845009863766, 0.5113143187961662, 0.6066785531441825], 
        [0.7082480686600264, 0.7264058872975387, 0.635154779102334, 0.5207791407472983, 0.6825721187970941]
    ])

    actual_output = sigmoid_layer.forward(input)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

sigmoid_tests.add_test(Test("Testing Sigmoid forward pass on 4x5 array", sigmoid_forward_test_2))

def sigmoid_backward_test_1():
    sigmoid_layer = Sigmoid()

    input = np.random.rand(1, 1)

    sigmoid_layer.forward(input)
    actual_grad_wrt_in = sigmoid_layer.backward(np.ones((1, 1)))

    def operation(inp):
        return np.sum(sigmoid_layer.forward(inp))
    
    expected_grad_wrt_in = nd.Gradient(operation)(input).reshape(input.shape)

    assert(expected_grad_wrt_in.shape == actual_grad_wrt_in.shape)
    np.testing.assert_allclose(expected_grad_wrt_in, actual_grad_wrt_in, rtol=0, atol=ABSOLUTE_TOLERANCE)

sigmoid_tests.add_test(Test("Testing Sigmoid backward pass on 1x1 array", sigmoid_backward_test_1))

def sigmoid_stacked_backward_test_1():
    sigmoid_layer_1 = Sigmoid()
    sigmoid_layer_2 = Sigmoid()
    sigmoid_layer_3 = Sigmoid()
    input = np.random.rand(1, 1)
    output = sigmoid_layer_1.forward(input)
    output = sigmoid_layer_2.forward(output)
    sigmoid_layer_3.forward(output)

    cur_grad_wrt_in = sigmoid_layer_3.backward(np.ones((1, 1)))
    cur_grad_wrt_in = sigmoid_layer_2.backward(cur_grad_wrt_in)
    actual_grad_wrt_in = sigmoid_layer_1.backward(cur_grad_wrt_in)

    def operation(inp):
        output = sigmoid_layer_1.forward(inp)
        output = sigmoid_layer_2.forward(output)
        return np.sum(sigmoid_layer_3.forward(output))
    
    expected_grad_wrt_in = nd.Gradient(operation)(input).reshape(input.shape)

    assert(expected_grad_wrt_in.shape == actual_grad_wrt_in.shape)
    np.testing.assert_allclose(expected_grad_wrt_in, actual_grad_wrt_in, rtol=0, atol=ABSOLUTE_TOLERANCE)

sigmoid_tests.add_test(Test("Testing stacked Sigmoid backward pass on 1x1 array", sigmoid_stacked_backward_test_1))

def sigmoid_backward_test_2():
    sigmoid_layer = Sigmoid()

    input = np.random.rand(4, 5)

    sigmoid_layer.forward(input)
    actual_grad_wrt_in = sigmoid_layer.backward(np.ones((4, 5)))

    def operation(inp):
        return np.sum(sigmoid_layer.forward(inp))
    
    expected_grad_wrt_in = nd.Gradient(operation)(input).reshape(input.shape)

    assert(expected_grad_wrt_in.shape == actual_grad_wrt_in.shape)
    np.testing.assert_allclose(expected_grad_wrt_in, actual_grad_wrt_in, rtol=0, atol=ABSOLUTE_TOLERANCE)

sigmoid_tests.add_test(Test("Testing Sigmoid backward pass on 4x5 array", sigmoid_backward_test_2))

def sigmoid_stacked_backward_test_2():
    sigmoid_layer_1 = Sigmoid()
    sigmoid_layer_2 = Sigmoid()
    sigmoid_layer_3 = Sigmoid()
    input = np.random.rand(4, 5)
    output = sigmoid_layer_1.forward(input)
    output = sigmoid_layer_2.forward(output)
    sigmoid_layer_3.forward(output)

    cur_grad_wrt_in = sigmoid_layer_3.backward(np.ones((4, 5)))
    cur_grad_wrt_in = sigmoid_layer_2.backward(cur_grad_wrt_in)
    actual_grad_wrt_in = sigmoid_layer_1.backward(cur_grad_wrt_in)

    def operation(inp):
        output = sigmoid_layer_1.forward(inp)
        output = sigmoid_layer_2.forward(output)
        return np.sum(sigmoid_layer_3.forward(output))
    
    expected_grad_wrt_in = nd.Gradient(operation)(input).reshape(input.shape)

    assert(expected_grad_wrt_in.shape == actual_grad_wrt_in.shape)
    np.testing.assert_allclose(expected_grad_wrt_in, actual_grad_wrt_in, rtol=0, atol=ABSOLUTE_TOLERANCE)

sigmoid_tests.add_test(Test("Testing stacked Sigmoid backward pass on 4x5 array", sigmoid_stacked_backward_test_2))

################################################# Testing Linear ##################################################

linear_tests = TestSuite("Testing Linear Layer")

def linear_forward_test_1(): 
    linear_layer = Linear(4, 5)

    input = np.array([
        [-0.6685841337060198, -0.7538826927644409, 0.04334270851242105, -0.3458545003688088]
    ])

    expected_output = np.array([
        [-1.1604393297969606, -0.977435196412541, 0.17858053969208637, -0.694503417025958, 2.2082602820795265]
    ])

    actual_output = linear_layer.forward(input)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

linear_tests.add_test(Test("Testing Linear forward pass on input w/ batch size 1", linear_forward_test_1))

def linear_forward_test_2():
    linear_layer = Linear(4, 5)

    input = np.array([
        [1.116572187262277, 0.6524266739045254, 0.23001030225256924, -0.29665570160728544], 
        [-0.8261004379220587, 0.7184794767899947, 0.4704364307796347, 1.7867131670116942], 
        [-0.9581397470491813, -0.6551066909182789, -0.8185628139806348, -0.6309136070045491]
    ])

    expected_output = np.array([
        [-0.7631502249720685, 4.402581855512059, 1.6948591508108317, 0.18005130044049103, -1.2163310416984658], 
        [0.8097987509395378, 0.46056582038468385, -2.680737885337229, 2.5786539808840225, 4.888070360686292], 
        [-0.9952807246425872, -1.24672571955961, 0.2583096166877682, -2.075824321029051, 2.9271722212471456]
    ])

    actual_output = linear_layer.forward(input)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

linear_tests.add_test(Test("Testing Linear forward pass on input w/ batch size 3", linear_forward_test_2))

def linear_backward_test_1():
    batch_size = 3
    input_dim = 4
    output_dim = 5

    linear_layer = Linear(input_dim, output_dim)
    input = np.random.rand(batch_size, input_dim)

    def operation(inp):
        return np.sum(linear_layer.forward(inp.reshape(input.shape)))
    
    expected_grad_wrt_in = nd.Gradient(operation)(input).reshape(input.shape)

    
    linear_layer.forward(input)
    actual_grad_wrt_in = linear_layer.backward(np.ones((batch_size, output_dim)))

    assert(expected_grad_wrt_in.shape == actual_grad_wrt_in.shape)
    np.testing.assert_allclose(expected_grad_wrt_in, actual_grad_wrt_in, rtol=0, atol=ABSOLUTE_TOLERANCE)

linear_tests.add_test(Test("Testing Linear backward pass grad w.r.t. input", linear_backward_test_1))

def linear_backward_test_2():
    batch_size = 3
    input_dim = 4
    output_dim = 5

    input = np.random.rand(batch_size, input_dim)
    W_custom = np.random.rand(output_dim, input_dim)
    linear_layer = Linear(input_dim, output_dim)
    _, b = linear_layer.getParams()
    linear_layer.setParams(W_custom, b)

    def operation(W):
        op_lin_layer = Linear(input_dim, output_dim)
        _, b = op_lin_layer.getParams()
        op_lin_layer.setParams(W.reshape((output_dim, input_dim)), b)
        return np.sum(op_lin_layer.forward(input)) / batch_size
    
    expected_grad_wrt_W = nd.Gradient(operation)(W_custom).reshape((output_dim, input_dim))

    
    linear_layer.forward(input)
    linear_layer.backward(np.ones((batch_size, output_dim)))
    actual_grad_wrt_W, _ = linear_layer.getGradients()

    assert(expected_grad_wrt_W.shape == actual_grad_wrt_W.shape)
    np.testing.assert_allclose(expected_grad_wrt_W, actual_grad_wrt_W, rtol=0, atol=ABSOLUTE_TOLERANCE)

linear_tests.add_test(Test("Testing Linear backward pass grad w.r.t. W", linear_backward_test_2))

def linear_backward_test_3():
    batch_size = 3
    input_dim = 4
    output_dim = 5
    
    input = np.random.rand(batch_size, input_dim)
    b_custom = np.random.rand(output_dim,)
    linear_layer = Linear(input_dim, output_dim)
    W, _ = linear_layer.getParams()
    linear_layer.setParams(W, b_custom)

    def operation(b):
        op_lin_layer = Linear(input_dim, output_dim)
        W, _ = op_lin_layer.getParams()
        op_lin_layer.setParams(W, b.reshape((output_dim)))
        return np.sum(op_lin_layer.forward(input)) / batch_size
    
    expected_grad_wrt_b = nd.Gradient(operation)(b_custom).reshape((output_dim,))

    
    linear_layer.forward(input)
    linear_layer.backward(np.ones((batch_size, output_dim)))
    _, actual_grad_wrt_b = linear_layer.getGradients()

    assert(expected_grad_wrt_b.shape == actual_grad_wrt_b.shape)
    np.testing.assert_allclose(expected_grad_wrt_b, actual_grad_wrt_b, rtol=0, atol=ABSOLUTE_TOLERANCE)

linear_tests.add_test(Test("Testing Linear backward pass grad w.r.t. b", linear_backward_test_3))

def linear_step_test():
    lin = Linear (4, 5)
    input = np.array([
        [-1.0485685331558576, -1.2424708563311841, -1.2666447798085965, -0.4780565721120151], 
        [-0.9128278254115503, -0.3473203105456596, 1.38680668484236, 1.0168810071584928], 
        [0.737346126782251, 0.5309997540702895, -1.5653182047898122, 0.10783205811626057]
    ])
    expected_W = np.array([
        [-0.7601629388918699, 1.2286887947724545, 0.2172913863688188, -0.02314784504323752], 
        [1.3143760275594145, 2.1433197587473267, 0.3807707530227747, -0.7837110006601617], 
        [0.24054857871110424, 0.7750805842233327, 0.5727535788278254, -1.9719104992769103], 
        [0.4716113962299077, -0.1444705789365526, 0.9427120567561578, 1.481568374491559], 
        [-2.4133314806686066, 0.6363230934157771, -0.1681437648101374, 0.6736482305426933]
    ])
    expected_b = np.array([-0.7742431063311538, 1.2151594900630993, 0.20247419870296532, -0.030992323399361726, 1.3002958601201307])

    lin.forward(input)
    lin.backward(np.ones((3, 5)))
    lin.step(0.01)

    actual_W, actual_b = lin.getParams()

    assert(expected_W.shape == actual_W.shape)
    assert(expected_b.shape == actual_b.shape)

    np.testing.assert_allclose(expected_W, actual_W, rtol=0, atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(expected_b, actual_b, rtol=0, atol=ABSOLUTE_TOLERANCE)

linear_tests.add_test(Test("Testing Linear backward pass step", linear_step_test))

############################################### Testing SoftmaxLoss ###############################################

softmaxloss_tests = TestSuite("Testing SoftmaxLoss Layer")

def softmaxloss_forward_test_1():
    sloss_layer = SoftmaxLoss()
    input = np.array([
        [0.3396077568873713, 0.11127275957881966, 0.166958383855476, 0.6059904329853792, 0.4395162707373098], 
        [0.9673695665895302, 0.9424160909917588, 0.4246891658340317, 0.512067110864167, 0.9446174864061644], 
        [0.5358407707277277, 0.757056793509038, 0.8629974809417822, 0.7806054988752749, 0.6888524423177692], 
        [0.8446294522058271, 0.6496048408074062, 0.5280344513005517, 0.5392682644683179, 0.5737299551874347]
    ])
    y = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ])

    expected_output = np.array([1.7915611577072965, 1.4524628423807384, 1.4773990479274293, 1.3989614248911193])

    actual_output = sloss_layer.forward(input, True, y=y)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

softmaxloss_tests.add_test(Test("Testing SoftmaxLoss forward with training = True", softmaxloss_forward_test_1))

def softmaxloss_forward_test_2():
    sloss_layer = SoftmaxLoss()
    input = np.array([
        [0.28711528569130784, 0.8023046511709442, 0.308725709044121, 0.8396521246009222, 0.7965904272924517], 
        [0.788104824218436, 0.6277260587161021, 0.5479822664168736, 0.19976206510926642, 0.6451534877619403], 
        [0.09358298178828028, 0.3034277555791959, 0.9447814767901875, 0.21047208618094182, 0.3944725904357873], 
        [0.03161166377826974, 0.8395808583665875, 0.09545987285015223, 0.5733456217590912, 0.25468587527137443]
    ])

    expected_output = np.array([
        [0.14088725630155116, 0.23583899209450568, 0.14396502572711714, 0.24481352755668911, 0.23449519832013682], 
        [0.24628142772551043, 0.20978771357498932, 0.1937080908195322, 0.13674695006851068, 0.21347581781145736], 
        [0.14182397287650883, 0.17493796664857184, 0.3322159323020028, 0.15940940472340345, 0.19161272344951313], 
        [0.13739705459852267, 0.30822934652719275, 0.14645572318226371, 0.2361834658899808, 0.1717344098020401]
    ])

    actual_output = sloss_layer.forward(input, False)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

softmaxloss_tests.add_test(Test("Testing SoftmaxLoss forward with training = False", softmaxloss_forward_test_2))

def softmaxloss_backward_test_1():
    sloss_layer = SoftmaxLoss()

    batch_size = 4
    num_classes = 5

    input = np.random.rand(batch_size, num_classes)
    random_labels = np.random.randint(0, num_classes, size=batch_size)
    y = np.zeros((batch_size, num_classes))
    y[np.arange(batch_size), random_labels] = 1

    def operation(inp):
        inp = inp.reshape(input.shape)
        return np.sum(sloss_layer.forward(inp, True, y=y))
    
    expected_grad_wrt_input = nd.Gradient(operation)(input).reshape(input.shape)
    sloss_layer.forward(input, True, y=y)
    actual_grad_wrt_input = sloss_layer.backward()

    assert(expected_grad_wrt_input.shape == actual_grad_wrt_input.shape)
    np.testing.assert_allclose(expected_grad_wrt_input, actual_grad_wrt_input, rtol=0, atol=ABSOLUTE_TOLERANCE)

softmaxloss_tests.add_test(Test("Testing SoftmaxLoss backward pass", softmaxloss_backward_test_1))

############################################ Testing NeuralNetwork_2HL ############################################

neuralnetwork_tests = TestSuite("Testing full Neural Network")

def neuralnetwork_forward_test_1():
    neural_network = NeuralNetwork_2HL()

    input = initialize_random_array((5, 784))
    y = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    ])

    expected_output = np.array([2.039630979595195, 3.1613593271474674, 13.016463350325699, 23.90224445029655, 23.258138872821206])

    actual_output = neural_network.forward(input, True, y=y)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

neuralnetwork_tests.add_test(Test("Testing NeuralNetwork_2HL forward pass with training = True", neuralnetwork_forward_test_1))

def neuralnetwork_forward_test_2():
    neural_network = NeuralNetwork_2HL()

    input = initialize_random_array((5, 784))

    expected_output = np.array([
        [0.10582685391176316, 0.13977803157313026, 0.13007670298043839, 0.6242420617601993, 6.344898925473497e-05, 4.679569501808883e-10, 2.6290926610701123e-09, 1.267771489305603e-05, 2.1960708922095711e-07, 3.6618224918436095e-10], 
        [0.9081651617367179, 0.04236810979766699, 0.0005986129792747834, 0.04790986567440679, 0.0009014063128534348, 1.5191542461711507e-09, 1.6521575632672584e-11, 5.6816520686014785e-05, 2.4674173419284348e-08, 7.685447669637201e-10], 
        [0.9249026133823518, 0.005221238831035597, 0.052416204694300125, 0.017398312621027808, 2.789269195450829e-05, 1.9720189001858937e-08, 1.944087784864468e-09, 3.133883272170013e-05, 2.223421459962816e-06, 1.5386087182977953e-07], 
        [0.9135059619650745, 0.005844531999503725, 0.029768088439164474, 0.050736099591031064, 3.456741090591828e-05, 1.5420114130465666e-09, 1.4187739635516047e-10, 0.0001107425979779221, 6.270825474646273e-09, 4.162815185324261e-11], 
        [0.10371646218454182, 0.004467187968527273, 0.012891044166909294, 0.8788309767119882, 1.3132110509982608e-06, 3.7654042124615554e-08, 7.927178329845245e-11, 9.288173642635331e-05, 9.627222397398375e-08, 1.5018308310578812e-11]
    ])

    actual_output = neural_network.forward(input, False)

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

neuralnetwork_tests.add_test(Test("Testing NeuralNetwork_2HL forward pass with training = False", neuralnetwork_forward_test_2))

def neuralnetwork_backward_test():
    neural_network = NeuralNetwork_2HL()
    input = initialize_random_array((5, 784))
    y = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    expected_grad_wrt_in_sum = -9.403725725747012

    neural_network.forward(input, True, y=y)
    actual_grad_wrt_in_sum = np.sum(neural_network.backward())

    assert(abs(actual_grad_wrt_in_sum - expected_grad_wrt_in_sum) < 1e-3)

neuralnetwork_tests.add_test(Test("Testing NeuralNetwork_2HL backward pass", neuralnetwork_backward_test))

def neuralnetwork_step_test():
    neural_network = NeuralNetwork_2HL()
    input = initialize_random_array((5, 784))
    y = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    for _ in range(1):
        neural_network.forward(input, True, y=y)
        neural_network.backward()
        neural_network.step(lr=0.001)

    actual_output = neural_network.forward(input, True, y=y)
    expected_output = np.array([0.46329760525871516, 2.9912559144376334, 10.443421508440363, 3.4801633568880495, 0.12506374624770186])

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

neuralnetwork_tests.add_test(Test("Testing NeuralNetwork_2HL step - one step", neuralnetwork_step_test))

def neuralnetwork_multistep_test():
    neural_network = NeuralNetwork_2HL()
    input = initialize_random_array((5, 784))
    y = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    for _ in range(10):
        neural_network.forward(input, True, y=y)
        neural_network.backward()
        neural_network.step(lr=0.001)

    actual_output = neural_network.forward(input, True, y=y)
    expected_output = np.array([0.4013500639634512, 2.583581519523094, 10.074227794801322, 3.1813196806237847, 0.09559122747726634])

    assert(expected_output.shape == actual_output.shape)
    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

neuralnetwork_tests.add_test(Test("Testing NeuralNetwork_2HL step - ten steps", neuralnetwork_multistep_test))

################################################ Testing train_NN #################################################

train_NN_tests = TestSuite("Testing SoftmaxLoss Layer")

def train_NN_test_1():
    nn = NeuralNetwork_2HL()
    train_NN(
        nn, 0.001, 10, 1, randomize=False, verbose=False
    )

    actual_output = nn.forward(initialize_random_array((1, 784)), False)

    expected_output = np.array([
        [0.2514889145584578, 0.3409029035315118, 0.2514137873617063, 0.15587466282046974, 0.00026127821928547654, 1.8092161690567012e-09, 7.398218328628999e-09, 5.767013086398041e-05, 7.72729529149794e-07, 1.4407410683669942e-09]
    ])

    np.testing.assert_allclose(expected_output, actual_output, rtol=0, atol=ABSOLUTE_TOLERANCE)

train_NN_tests.add_test(Test("Testing train_NN neural network update", train_NN_test_1))

def train_NN_test_2():
    nn = NeuralNetwork_2HL()
    actual_train_loss, _, _, _ = train_NN(
        nn, 0.001, 10, 1, randomize=False, verbose=False
    )

    expected_train_loss = np.array([10.512333964407812])

    np.testing.assert_allclose(np.array(expected_train_loss), actual_train_loss, rtol=0, atol=ABSOLUTE_TOLERANCE)

train_NN_tests.add_test(Test("Testing train_NN neural network train losses", train_NN_test_2))

def train_NN_test_3():
    nn = NeuralNetwork_2HL()
    _, actual_val_loss, _, _ = train_NN(
        nn, 0.001, 10, 1, randomize=False, verbose=False
    )

    expected_val_loss = np.array([9.942916533841544])

    np.testing.assert_allclose(expected_val_loss, actual_val_loss, rtol=0, atol=ABSOLUTE_TOLERANCE)

train_NN_tests.add_test(Test("Testing train_NN neural network validation losses", train_NN_test_3))

def train_NN_test_4():
    nn = NeuralNetwork_2HL()
    _, _, actual_train_acc, _ = train_NN(
        nn, 0.001, 10, 1, randomize=False, verbose=False
    )

    expected_train_acc = np.array([0.10666666666666655])

    np.testing.assert_allclose(expected_train_acc, actual_train_acc, rtol=0, atol=ABSOLUTE_TOLERANCE)

train_NN_tests.add_test(Test("Testing train_NN neural network train accuracies", train_NN_test_4))

def train_NN_test_5():
    nn = NeuralNetwork_2HL()
    _, _, _, actual_val_acc = train_NN(
        nn, 0.001, 10, 1, randomize=False, verbose=False
    )

    expected_val_acc = np.array([0.08500000000000002])

    np.testing.assert_allclose(expected_val_acc, actual_val_acc, rtol=0, atol=ABSOLUTE_TOLERANCE)

train_NN_tests.add_test(Test("Testing train_NN neural network train accuracies", train_NN_test_5))


