#!/usr/bin/env python

import unittest
import diffractionimaging as di
import numpy as np

class TestClusterSize(unittest.TestCase):

    def setUp(self):
        pass

    def test_example(self):
        trueparams = (1, 30)
        x = np.linspace(-200,200,401)  # use pixel as unit
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        img = di.clustersize.diffraction_sphere(r, *trueparams)
        params = di.autofit_sphere(img)[0]
        self.assertAlmostEqual(trueparams[0], params[0], 2)
        self.assertAlmostEqual(trueparams[1], params[1], 1)


if __name__ == '__main__':
    unittest.main()
