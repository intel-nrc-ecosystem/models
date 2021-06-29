###############################################################
# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2018-2021 Intel Corporation.

# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.

# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
###############################################################

import os
import unittest

import matplotlib as mpl

haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')
from nxsdk_modules.csp.src.csp_solver_src import *


class TestOptimizationSolver(unittest.TestCase):

    def setUp(self) -> None:
        self.osolver = OptimizationSolver(problem=())

    def tearDown(self) -> None:
        pass

    def test_collected_solution(self):
        self.assertEqual(0, len(self.osolver.collected_solutions))
        for e in [1, 2, 3]:
            self.osolver._collected_solutions.append(e)
            self.assertEqual(e, self.osolver.collected_solutions[-1])
        self.assertEqual(3, len(self.osolver.collected_solutions))


class TestCspSolver(unittest.TestCase):

    def setUp(self) -> None:
        self.relation = np.asarray([[0, 1],
                                    [1, 0]])
        self.csp = (3, 2, [(0, 1, np.logical_not(np.eye(2))),
                           (1, 2, np.logical_not(self.relation))])
        self.csp_solver = CspSolver(problem=self.csp, snnkwargs=dict(w_constraints_inh=-13, w_wta_inh=-10, w_exp=2))

    def tearDown(self) -> None:
        os.system("scancel -u %s"%os.getlogin())

    def test_classes(self):
        self.assertIsInstance(self.csp_solver, CspSolver)
        self.assertIsInstance(self.csp_solver, OptimizationSolver)

    def test_problem_input(self):
        self.assertTupleEqual(self.csp, self.csp_solver.problem)
        self.assertEqual(3, self.csp_solver.csp.num_variables)
        self.assertEqual(2, self.csp_solver.csp.domain_size)
        self.assertEqual(self.csp[-1], self.csp_solver.csp.constraints)
        val = np.logical_not(self.csp_solver.csp.constraints[-1][2])
        self.assertTrue((self.relation == val).all())

    def test_node_constraints(self):
        csp = (3, 2, [(1, 2, np.eye(2)),
                      (1, 2, self.relation)])
        csp_solver = CspSolver(problem=csp)
        clamps = [(0, 1), (1, 0)]
        csp_solver.node_constraints = clamps
        self.assertEqual(clamps, csp_solver.node_constraints)
        solution = csp_solver.solve(runtime=50)
        print(solution)

    def test_solve(self):
        solution = self.csp_solver.solve(runtime=100, vr_low=-20)
        print(solution)


    def test_solve_sudoku(self):
        from nxsdk_modules.csp.src.translators.latin2csp import translateSudoku
        params = {2: {'w_exp': 5, 'vth': 10000, 'seed': 4200177969},
                  3: {'w_exp': 5, 'vth': 5000, 'seed': 2837769982},
                  4: {'w_exp': 5, 'vth': 5000, 'seed': 699767509},
                  5: {'w_exp': 5, 'vth': 5000, 'seed': 459378927},
                  6: {'w_exp': 5, 'vth': 5000, 'seed': 2684185586},
                  7: {'w_exp': 5, 'vth': 5000, 'seed': 2150499300},
                  8: {'w_exp': 5, 'vth': 5000, 'seed': 2231247799},
                  9: {'w_exp': 5, 'vth': 6000, 'seed': 3346769215},
                  10: {'w_exp': 5, 'vth': 6000, 'seed': 3346769215},
                  11: {'w_exp': 5, 'vth': 6000, 'seed': 3658386454},
                  12: {'w_exp': 5, 'vth': 6000, 'seed': 1943687762},
                  }
        for size in range(2, 13):
            print("Running for Size", size)
            os.system("scancel -u %s" % os.getlogin())
            game = translateSudoku(np.zeros((size, size)), is_latin=True)
            csp = (game.number_of_variables, game.states_per_element, (game.constraints, np.logical_not(np.eye(size))))
            p = params[size]
            csp_solver = CspSolver(problem=csp,
                                   snnkwargs=dict(w_constraints_inh=-12, w_wta_inh=-12, w_exp=p['w_exp']),
                                   nkwargs=dict(v_th_1_mant=p['vth']))
            solution = csp_solver.solve(seed=p["seed"],
                                        runtime=300000, randomize_vinit=True, set_random_initial_state=False)
            solved = csp_solver._check_solution(solution) if solution is not None else False
            if solved:
                print(solution.reshape(size, size))


class TestQuboSolver(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        os.system("scancel -u %s" % os.getlogin())

    def test_userx(self):
        """Polynomial minimization with y=-5x_1 -3x_2 -8x_3 -6x_4 + 4x_1x_2+8x_1x_3+2x_2x_3+10x_3x_4"""
        for trial in range(5):
            os.system("scancel -u %s" % os.getlogin())
            Q = 2 * np.asarray([[-5, 2, 4, 0],
                                [2, -3, 1, 0],
                                [4, 1, -8, 5],
                                [0, 0, 5, -6]])
            solver = QuboSolver(Q, q_weight_scaling=1, snnkwargs=dict(w_exp=6, box_duration=6),
                                nkwargs=dict(noise_at_multicompartment=dict(mantissa=0, exponent=1),
                                             v_th_1_mant=8 * 2 * 6,
                                             bias_exp=6))
            solution = solver.solve(runtime=200, seed=np.random.random_integers(0, 2 ** 32, 1)[0],
                                    target_cost=2,
                                    )
            print(solution + 1)

    def test_max_cut_problem(self):
        """Max-Cut Problem"""
        solutions = []
        for trial in range(10):
            os.system("scancel -u %s" % os.getlogin())
            Q = 4 * np.asarray([[2, -1, -1, 0, 0],
                                [-1, 2, 0, -1, 0],
                                [-1, 0, 3, -1, -1],
                                [0, -1, -1, 3, -1],
                                [0, 0, -1, -1, 2]])
            solver = QuboSolver(-Q, q_weight_scaling=2, snnkwargs=dict(w_exp=6, box_duration=6),
                                nkwargs=dict(noise_at_multicompartment=dict(mantissa=0, exponent=1),
                                             v_th_1_mant=3 * 4 * 6,
                                             bias_exp=6))
            solution = solver.solve(runtime=20000, seed=np.random.random_integers(0, 2 ** 32, 1)[0],
                                    target_cost=2,
                                    )
            print(solution + 1)

            solutions.append(solution + 1)
            for sol in solutions:
                print(sol)

    def test_set_packing_problem(self):
        """Set Packing Problem."""
        solutions = []
        for trial in range(10):
            os.system("scancel -u %s" % os.getlogin())
            Q = 4 * np.asarray([[1, -3, -3, -3],
                                [-3, 1, 0, 0],
                                [-3, 0, 1, -3],
                                [-3, 0, -3, 1]])
            solver = QuboSolver(-Q, q_weight_scaling=1, snnkwargs=dict(w_exp=6, box_duration=6),
                                nkwargs=dict(noise_at_multicompartment=dict(mantissa=0, exponent=7),
                                             v_th_1_mant=1 * 4 * 6,
                                             bias_exp=6))
            solution = solver.solve(runtime=20000, seed=np.random.random_integers(0, 2 ** 32, 1)[0],
                                    target_cost=2,
                                    )
            print(solution + 1)

            solutions.append(solution + 1)
            for sol in solutions:
                print(sol)

    def test_map_coloring(self):
        """Map coloring."""
        solutions = []
        for trial in range(10):
            os.system("scancel -u %s" % os.getlogin())
            Q = 4 * np.asarray([[-4., 4., 4., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
                                [4., -4., 4., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.],
                                [4., 4., -4., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2.],
                                [2., 0., 0., -4., 4., 4., 2., 0., 0., 2., 0., 0., 2., 0., 0.],
                                [0., 2., 0., 4., -4., 4., 0., 2., 0., 0., 2., 0., 0., 2., 0.],
                                [0., 0., 2., 4., 4., -4., 0., 0., 2., 0., 0., 2., 0., 0., 2.],
                                [0., 0., 0., 2., 0., 0., -4., 4., 4., 2., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 2., 0., 4., -4., 4., 0., 2., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 2., 4., 4., -4., 0., 0., 2., 0., 0., 0.],
                                [0., 0., 0., 2., 0., 0., 2., 0., 0., -4., 4., 4., 2., 0., 0.],
                                [0., 0., 0., 0., 2., 0., 0., 2., 0., 4., -4., 4., 0., 2., 0.],
                                [0., 0., 0., 0., 0., 2., 0., 0., 2., 4., 4., -4., 0., 0., 2.],
                                [2., 0., 0., 2., 0., 0., 0., 0., 0., 2., 0., 0., -4., 4., 4.],
                                [0., 2., 0., 0., 2., 0., 0., 0., 0., 0., 2., 0., 4., -4., 4.],
                                [0., 0., 2., 0., 0., 2., 0., 0., 0., 0., 0., 2., 4., 4., -4.], ])

            solver = QuboSolver(Q, q_weight_scaling=1, snnkwargs=dict(w_exp=6, box_duration=6),
                                nkwargs=dict(noise_at_multicompartment=dict(mantissa=0, exponent=7),
                                             v_th_1_mant=4 * 4 * 6,
                                             bias_exp=6))
            solution = solver.solve(runtime=20000, seed=np.random.random_integers(0, 2 ** 32, 1)[0],
                                    target_cost=5,
                                    )
            print(solution + 1)

            solutions.append(solution + 1)
            for sol in solutions:
                print(sol)

    def test_set_partitioning(self):
        """Set partitioning."""
        solutions = []
        for trial in range(10):
            os.system("scancel -u %s" % os.getlogin())
            Q = np.asarray([[-17., 10., 10., 10., 0., 20.],
                            [10., -18., 10., 10., 10., 20.],
                            [10., 10., -29., 10., 20., 20.],
                            [10., 10., 10., -19., 10., 10.],
                            [0., 10., 20., 10., -17., 10.],
                            [20., 20., 20., 10., 10., -28.]])
            solver = QuboSolver(Q, q_weight_scaling=10, snnkwargs=dict(w_exp=6, box_duration=6),
                                nkwargs=dict(noise_at_multicompartment=dict(mantissa=0, exponent=3),
                                             v_th_1_mant=2 * 2 * 6,
                                             bias_exp=6))
            solution = solver.solve(runtime=20000, seed=np.random.random_integers(0, 2 ** 32, 1)[0],
                                    target_cost=2,
                                    )
            print(solution + 1)

            solutions.append(solution + 1)
            for sol in solutions:
                print(sol)


class TestCspAdjMat(unittest.TestCase):
    """Tests the adjacency matrix for the CSP solver is built adequately including WTA and constraints."""

    def test_adj_mtx(self):
        am = CspAdjacencyMatrix(num_variables=3, domain_size=3, constraints=[(1, 2, np.eye(3))])
        if haveDisplay:
            plt.imshow(am.adjacency_mtx)
            plt.show()


class TestCspPrototypeMap(unittest.TestCase):
    def test_prototype_map(self):
        pm = CspPrototypeMap(number_of_variables=4,
                             states_per_variable=3,
                             clamped_values=[(1, 2), (3, 0)])
        print(pm.prototype_map)


class TestMulticompartment(unittest.TestCase):
    """Test multicompartment design for CSP solver with online validation.

    In the text names mc refers to multicompartment and c0, c1, c2 and c3 to the compartments on the mc,
    with c0 being the principal compartment which integrates synaptic input from other rincipal compartments and
    sends spikes to them. u, v and s refer to current, voltage and spikes respectively and can be indexed aswell,
    e.g, u2 is the current variables of c2.
    """

    def setUp(self) -> None:
        self.mckwargs = dict(noise_at_multicompartment=None,
                             enable_noise=False,
                             vth_1_mant=1000,
                             )
        self.snn = None
        self.multicompartment = MultiCompartment()

    def tearDown(self) -> None:
        os.system("scancel -u %s"%os.getlogin())

    def test_userx_probes(self):
        mc = MultiCompartment()
        mc.probe('u', 's', 'v')
        mc.run(30)
        cls = ['g', 'r', 'b', 'k']
        for var in ['u', 's', 'v']:
            for idx in [None, 0, 1, 2]:
                mc.plot_probe(var, linecolor=cls if idx is None else cls[idx], index=idx)
                plt.show()

    def test_create_multicompartment(self):
        multicompartment = MultiCompartment()
        self.assertIsInstance(multicompartment, MultiCompartment)

    def test_probes(self):
        self.multicompartment.probe('v', 's', 'u', index=None)
        self.multicompartment.run(50)
        self.multicompartment.plot_probe('v')
        self.multicompartment.plot_probe('u')
        self.multicompartment.plot_probe('s')
        plt.show()

    def test_firing_rate_of_c0_without_noise(self):
        multicompartment = MultiCompartment(
            bias_to_fire=10,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe = multicompartment.probe('s', index=0)
        multicompartment.run(101)
        self.assertEqual(9, sum(sprobe.data))

    def test_noise_amplitud(self):
        na = [3, 7, 15, 31, 63, 127, 254, 255, 510, 1020, 2040, 4080, 8160, 16320, 32640, 65280]
        for exp in range(9):
            multicompartment = MultiCompartment(
                bias_to_fire=-1,
                enable_noise=True,
                noise_at_multicompartment={'mantissa': 0, 'exponent': exp},
                v_th_1_mant=5000,
                randomized_seeds=False,
                randomize_v_init=False,
            )
            vprobe = multicompartment.probe('v', index=0)
            # set decay to max so that the noise distribution does not get affected by accumulation effects
            multicompartment.v_decay = 2 ** 12 - 1
            multicompartment.run(100)
            self.assertTrue(0.5 * multicompartment.noise_amplitude < max(vprobe.data) - min(vprobe.data)
                            <= multicompartment.noise_amplitude)

    def test_effect_of_input_weight_on_u0(self):
        multicompartment = MultiCompartment(
            bias_to_fire=10,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        uprobe = multicompartment.probe('u', index=0)
        # create auxiliary spike generator to test functionality.
        sg = 2
        spikeGenerator = multicompartment.main_net.createSpikeGenProcess(numPorts=1)
        spikeGenerator.addSpikes(0, spikeTimes=[sg])
        # connect spike generator to compartment
        connProto = nx.ConnectionPrototype(weight=-6,
                                           weightExponent=0,
                                           delay=2,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
        spikeGenerator.connect(multicompartment.compartment_group[0], connProto)
        multicompartment.run(8)
        self.assertEqual([0] * sg + [-6 * 2 ** 6] * connProto.delay + [0] * (8 - sg - connProto.delay), uprobe.data)

    def test_box_psps_in_u1_after_c0_spikes(self):
        multicompartment = MultiCompartment(
            bias_to_fire=10,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        uprobe = multicompartment.probe('u', index=1)
        multicompartment.run(20)
        s0 = multicompartment.bias_to_fire
        self.assertListEqual(list(range(s0 + 1, s0 + 1 + multicompartment.box_duration)),
                             np.where(np.asarray(uprobe.data) != 0)[0].tolist())

    def test_u1_remains_0_if_c0_does_not_spike(self):
        multicompartment = MultiCompartment(
            bias_to_fire=-1,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe = multicompartment.probe('s', index=0)
        uprobe = multicompartment.probe('u', index=1)
        multicompartment.run(20)
        assert sum(sprobe.data) == 0
        self.assertFalse(np.asarray(uprobe.data).any())

    def test_c2_spikes_if_c1_spikes_and_u0_is_0(self):
        multicompartment = MultiCompartment(
            bias_to_fire=10,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe_c1 = multicompartment.probe('s', index=1)
        sprobe_c2 = multicompartment.probe('s', index=2)
        uprobe_c0 = multicompartment.probe('u', index=0)
        multicompartment.run(20)
        assert sprobe_c1.data[multicompartment.bias_to_fire + 1] == 1
        assert not np.asarray(uprobe_c0.data).any()
        condition = np.where(np.asarray(sprobe_c1.data) & ~np.asarray(uprobe_c0.data))
        self.assertTrue(np.asarray(sprobe_c2.data)[condition].all())

    def test_c2_spikes_for_duration_of_box_after_s0(self):
        multicompartment = MultiCompartment(
            bias_to_fire=10,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe_c0 = multicompartment.probe('s', index=0)
        sprobe_c1 = multicompartment.probe('s', index=1)
        multicompartment.run(30)
        assert sprobe_c1.data[multicompartment.bias_to_fire + 1] == 1
        expected = np.asarray(sprobe_c0.data)
        for timestep in range(multicompartment.box_duration - 1):
            expected[1:] |= expected[:-1]
        tested = np.asarray(sprobe_c1.data)
        self.assertTrue((expected[:-1] == tested[1:]).all())

    def test_c2_does_not_spike_if_c1_spikes_and_u0_is_not_0(self):
        multicompartment = MultiCompartment(
            bias_to_fire=10,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )

        # create auxiliary spike generator to test functionality.
        sg = multicompartment.bias_to_fire + 1
        spikeGenerator = multicompartment.main_net.createSpikeGenProcess(numPorts=1)
        spikeGenerator.addSpikes(0, spikeTimes=[sg])
        connProto = nx.ConnectionPrototype(weight=-6,
                                           weightExponent=6,
                                           delay=6,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
        spikeGenerator.connect(multicompartment.compartment_group[0], connProto)

        sprobe_c1 = multicompartment.probe('s', index=1)
        sprobe_c2 = multicompartment.probe('s', index=2)
        uprobe_c0 = multicompartment.probe('u', index=0)
        multicompartment.run(20)
        assert np.asarray(sprobe_c1.data[sg:sg + 6]).all()
        assert np.asarray(uprobe_c0.data[sg:sg + 6]).all()
        condition = np.where(np.asarray(sprobe_c1.data) & np.asarray(uprobe_c0.data))
        self.assertFalse(np.asarray(sprobe_c2.data)[condition].any())

    def test_c2_does_not_spike_if_c1_does_not_spikes_and_u0_is_not_0(self):
        multicompartment = MultiCompartment(
            bias_to_fire=-1,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )

        # create auxiliary spike generator to test functionality.
        sg = 10
        spikeGenerator = multicompartment.main_net.createSpikeGenProcess(numPorts=1)
        spikeGenerator.addSpikes(0, spikeTimes=[sg])
        connProto = nx.ConnectionPrototype(weight=-6,
                                           weightExponent=6,
                                           delay=6,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
        spikeGenerator.connect(multicompartment.compartment_group[0], connProto)

        sprobe_c1 = multicompartment.probe('s', index=1)
        sprobe_c2 = multicompartment.probe('s', index=2)
        uprobe_c0 = multicompartment.probe('u', index=0)
        multicompartment.run(20)
        assert not np.asarray(sprobe_c1.data[:]).any()
        assert np.asarray(uprobe_c0.data[sg:sg + 6]).all()
        condition = np.where(~np.asarray(sprobe_c1.data) & np.asarray(uprobe_c0.data))
        self.assertFalse(np.asarray(sprobe_c2.data)[condition].any())

    def test_c2_does_not_spike_if_c1_does_not_spikes_and_u0_is_0(self):
        multicompartment = MultiCompartment(bias_to_fire=-1,
                                            enable_noise=False,
                                            randomized_seeds=False,
                                            randomize_v_init=False,
                                            )
        sprobe_c1 = multicompartment.probe('s', index=1)
        sprobe_c2 = multicompartment.probe('s', index=2)
        uprobe_c0 = multicompartment.probe('u', index=0)
        multicompartment.run(50)
        assert not np.asarray(sprobe_c1.data[:]).any()
        assert not np.asarray(uprobe_c0.data).any()
        self.assertFalse(np.asarray(sprobe_c2.data).any())

    def test_u2_remains_0_if_c0_does_not_emite_spikes(self):
        multicompartment = MultiCompartment(
            bias_to_fire=-1,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe_c0 = multicompartment.probe('s', index=0)
        uprobe_c2 = multicompartment.probe('u', index=2)
        multicompartment.run(50)
        assert not np.asarray(sprobe_c0.data[:]).any()
        self.assertFalse(np.asarray(uprobe_c2.data).any())

    def test_effect_of_input_weight_on_u2(self):
        w_mant = -6
        w_exp = 6
        multicompartment = MultiCompartment(bias_to_fire=10,
                                            enable_noise=False,
                                            randomized_seeds=False,
                                            v_th_1_mant=400,
                                            randomize_v_init=False,
                                            w_min=w_mant * 2 ** 6,
                                            )
        # create auxiliary spike generator to test functionality.
        sg = multicompartment.bias_to_fire + 3
        spikeGenerator = multicompartment.main_net.createSpikeGenProcess(numPorts=1)
        spikeGenerator.addSpikes(0, spikeTimes=[sg])
        connProto = nx.ConnectionPrototype(weight=w_mant,
                                           weightExponent=w_exp,
                                           delay=6,
                                           postSynResponseMode=nx.SYNAPSE_POST_SYN_RESPONSE_MODE.BOX)
        spikeGenerator.connect(multicompartment.compartment_group[0], connProto)

        sprobe_c1 = multicompartment.probe('s', index=1)
        uprobe_c0 = multicompartment.probe('u', index=0)
        vprobe_c2 = multicompartment.probe('v', index=2)
        multicompartment.run(20)

        assert sprobe_c1.data[sg] == 1
        assert uprobe_c0.data[sg] != 0
        self.assertEqual(uprobe_c0.data[sg], vprobe_c2.data[sg] - multicompartment.compartment_group[
            2].bias - multicompartment.compartment_group[0].bias)

    def test_c3_does_not_fire_intrinsically(self):
        multicompartment = MultiCompartment(
            bias_to_fire=-1,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe_c1 = multicompartment.probe('s', index=1)
        sprobe_c3 = multicompartment.probe('s', index=3)
        multicompartment.run(20)
        assert not np.asarray(sprobe_c1.data).any()
        self.assertFalse(np.asarray(sprobe_c3.data).any())

    def test_c3_replicates_c2_spikes_with_delay_1(self):
        multicompartment = MultiCompartment(
            bias_to_fire=6,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        sprobe_c2 = multicompartment.probe('s', index=2)
        uprobe_c3 = multicompartment.probe('u', index=3)
        multicompartment.run(20)
        assert sprobe_c2.data[multicompartment.bias_to_fire + 1] == 1
        self.assertTrue((np.asarray(uprobe_c3.data[1:]) == 128 * np.asarray(sprobe_c2.data[:-1])).all())

    def test_assertion_rised_when_btf_smaller_than_rho(self):
        with self.assertRaises(AssertionError):
            MultiCompartment(
                bias_to_fire=4,
                enable_noise=False,
                randomized_seeds=False,
                randomize_v_init=False,
            )

    def test_number_of_synapses_in_mc(self):
        multicompartment = MultiCompartment(
            bias_to_fire=6,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        # connection from c0 to c1 passing s(t-1) plus that from c2 to c3 to pass z(t-1) to be read by host.
        self.assertEqual(2, multicompartment.main_net.numConnections)

    def test_number_of_compartments_in_mc(self):
        multicompartment = MultiCompartment(
            bias_to_fire=6,
            enable_noise=False,
            randomized_seeds=False,
            randomize_v_init=False,
        )
        # this is a 4 compartment neuron.
        self.assertEqual(4, multicompartment.main_net.numCompartments)


class TestSummationNeuron(unittest.TestCase):
    """Test configuration and parameters for summation neuron.
    """

    def setUp(self) -> None:
        self.summation_neuron = SummationNeuron(sigma_threshold=6, _num_summation_neurons=2)

    def test_probes(self):
        self.summation_neuron.probe('v', 's', 'u', index=None)
        self.summation_neuron.run(20)
        self.summation_neuron.plot_probe('v')
        self.summation_neuron.plot_probe('u')
        self.summation_neuron.plot_probe('s')
        plt.show()


if __name__ == '__main__':
    unittest.main()
