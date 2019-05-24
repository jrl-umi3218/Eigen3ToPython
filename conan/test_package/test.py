#!/usr/bin/env python
# -*- coding: utf-8 -*-

import eigen
print("Eigen version: %s" % eigen.EigenVersion())
print("Random Vector3d: %s" % eigen.Vector3d.Random().transpose())
