import numpy as np
import os
import math
import xml.etree.ElementTree as ET
import mujoco

"""在mujocoo中对斯图尔特平台进行构型变化"""

def generate_model(
    body_parameter_scale: np.array,
    save_path: str,
    generations: float,
    damping_scale: float = 1.0,
):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, str(generations + 1)+".xml")
    mesh_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xml", "stl")).replace("\\", "/")
    parallel_default = [1, 1, 1, 1, 1,
                        0.09742, 0.188, 0.220, # 位置
                        0.052, 0.01, 0.007,  # 质量
                        0.01871796, 0.03525102, 0.03871796, 0.00061102, -0.03988726, -0.00141528, # site的位置
                        -0.01988814, 0.03322523, 0.02116929, -0.03383574, -0.01882983, -0.03383625,
                        ]
    num_param = len(parallel_default)
    parallel_robot_xml ='''
    <mujoco model="parallelrobot">
        <option solver="CG" tolerance="1e-7" timestep="0.001" integrator="implicit"/>
        <compiler meshdir="{MESH_DIR}"/>

        <asset>
            <mesh file="dong1.STL" scale="{1} {1} {3}" />
            <mesh file="dong2.STL" scale="{1} {1} {3}" />
            <mesh file="hinge.STL" />
            <mesh file="jing.STL" scale="{2} {2} {4}" />
            <mesh file="ten.STL" />
            <mesh file="up.STL" scale="{5} {5} 1" />
            <mesh file="down.STL" />
            <texture name="groundplane" type="2d"  builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" mark="edge" markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            <material name="hinge" rgba="0.412 0.412 0.412 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="top"   rgba="0.704 0.502 0.247 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="down"  rgba="0.447 0.455 0.447 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="dong2" rgba="0.75 0.75 0.75 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="jing"  rgba="0.275 0.51 0.706 1" reflectance="1" specular="1" shininess="0.5"/>
            <material name="dong1" rgba="0.275 0.51 0.706 1" reflectance="0.8" specular="1" shininess="0.5"/>
        </asset>
        
        <default>
            <default class="leg-slide">
                <joint type="slide" range="-0.5 0.5" axis="0 0 1" damping="0.15" stiffness="0" armature="0.01" />
            </default>
            <default class="hin">
                <joint type="hinge" limited="false" damping="0.15" stiffness="0" armature="0.01" />
            </default>
        </default>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" pos="0 0 0" size="0 0 0.1" type="plane" material="groundplane"/>

            <body name="top" pos="0 0 0.265">
                <freejoint/>
                <geom name="top_geom" type="mesh" mesh="up" material="top" contype="1" conaffinity="1" margin="-0.025" mass="0.173" solref="0.02 1" solimp="0.8 0.9 0.01"/>
                <site name="con1" type="sphere" pos = "{12} {13} -0.005" size="0.008 0.008 0.008"/>
                <site name="con2" type="sphere" pos = "{14} {15} -0.005" size="0.008 0.008 0.008"/>
                <site name="con3" type="sphere" pos = "{16} {17} -0.005" size="0.008 0.008 0.008"/>
                <site name="con4" type="sphere" pos = "{18} {19} -0.005" size="0.008 0.008 0.008"/>
                <site name="con5" type="sphere" pos = "{20} {21} -0.005" size="0.008 0.008 0.008"/>
                <site name="con6" type="sphere" pos = "{22} {23} -0.005" size="0.008 0.008 0.008"/>

                <site name="circle_center" pos="0 0 -0.005" size="0.001"/>
            </body>

            <body name="down" pos="0 0 0">
                <geom type="mesh" mesh="down" pos="0 0 0.015"  material="down" mass="0.623" />
            </body>

            <replicate count="3" euler="0 0 120">
                <body name="leg0" pos="0.07436 0.06631 0.02"> 
                    <geom type="mesh" mesh="hinge" pos="0 0 0" material="hinge" mass="0.007" contype="0" conaffinity="0" condim="1"/>
                    <geom name="ten0" type="mesh" mesh="ten" pos="0 0 0.01" mass="0.00155" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                    <body name="legdown0" pos="0 0 0.01" euler="8 -14 0">               
                        <joint type="hinge" axis="1 0 0" class="hin"/>
                        <joint type="hinge" axis="0 1 0" class="hin"/>
                        <geom type="mesh" mesh="hinge" pos="0 0 0.01" mass="0.007" euler="180 0 90" material="hinge" contype="0" conaffinity="0" />
                        <geom type="mesh" mesh="jing" pos="0 0 0.02" euler="0 0 0" material="jing" mass="{9}"/>
                        <body name="legup0" pos="0 0 0.01" euler="0 0 0">
                            <joint name="slide0" class="leg-slide"/>
                            <joint type="hinge" axis="0 0 1" limited="false" damping="0.5" armature="0.01"/>
                            <geom type="mesh" mesh="dong1" pos="0 0 {6}" euler="0 0 0" material="dong1" contype="1" conaffinity="1" margin="-0.1" mass="{10}" />
                            <geom type="mesh" mesh="dong2" pos="0 0 {7}" euler="0 0 0" material="dong2" contype="1" conaffinity="1" mass="{11}" />
                            <site name="conc0" type="sphere" pos = "0 0 {8}" size="0.008 0.008 0.008" />
                        </body>
                    </body>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                </body>

                <body name="leg1" pos="0.09436 0.03167 0.02">
                    <geom type="mesh" mesh="hinge" pos="0 0 0" mass="0.007" material="hinge" contype="0" conaffinity="0"/>
                    <geom name="ten1" type="mesh" mesh="ten" pos="0 0 0.01" mass="0.00155" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                    <body name="legdown1" pos="0 0 0.01" euler="8 -14 0">
                        <joint type="hinge" axis="1 0 0" class="hin"/>
                        <joint type="hinge" axis="0 1 0" class="hin"/>
                        <geom type="mesh" mesh="hinge" pos="0 0 0.01" mass="0.007" euler="180 0 90" material="hinge" contype="0" conaffinity="0"/>
                        <geom type="mesh" mesh="jing" pos="0 0 0.02" euler="0 0 0" material="jing" mass="{9}" />
                        <body name="legup1" pos="0 0 0.01" euler="0 0 0">
                            <joint name="slide1" class="leg-slide"/> 
                            <joint type="hinge" axis="0 0 1" limited="false" damping="0.5" armature="0.01"/>
                            <geom type="mesh" mesh="dong1" pos="0 0 {6}" euler="0 0 0" material="dong1" contype="1" conaffinity="1" margin="-0.1" mass="{10}" />
                            <geom type="mesh" mesh="dong2" pos="0 0 {7}" euler="0 0 0" material="dong2" contype="1" conaffinity="1" mass="{11}" />
                            <site name="conc1" type="sphere" pos = "0 0 {8}" size="0.008 0.008 0.008"  />
                        </body>
                    </body>
                </body>
            </replicate>
        </worldbody>

        <actuator>
            <motor name="a0" joint="slide00" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a1" joint="slide10" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a2" joint="slide01" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a3" joint="slide11" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a4" joint="slide02" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a5" joint="slide12" gear="1" damping="0.5" ctrlrange="-10 10"/>
        </actuator>

        <equality>
            <connect site1="con1" site2="conc00" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con2" site2="conc10" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con3" site2="conc01" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con4" site2="conc11" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con5" site2="conc02" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con6" site2="conc12" solref="0.001 1" solimp="0.99 0.99 0.001"/>
        </equality>
        
        <sensor>
            <framepos name="center_pos" objtype="site" objname="circle_center"/>
            <framequat name="center_quat" objtype="site" objname="circle_center"/>
            <framelinvel name="center_linvel" objtype="site" objname="circle_center"/>
            <frameangvel name="center_angvel" objtype="site" objname="circle_center"/>
            <framelinacc name="center_linacc" objtype="site" objname="circle_center"/>
            <frameangacc name="center_angacc" objtype="site" objname="circle_center"/>
        </sensor>

    </mujoco>'''

    parallel_robot_xml = parallel_robot_xml.replace("{MESH_DIR}", mesh_dir)
    parallel_robot_xml = parallel_robot_xml.replace(
        'damping="0.15"', f'damping="{0.15 * damping_scale:g}"'
    )
    parallel_robot_xml = parallel_robot_xml.replace(
        'damping="0.5"', f'damping="{0.5 * damping_scale:g}"'
    )
    for i in range(num_param):
        parallel_robot_xml = parallel_robot_xml.replace("{"+str(i+1)+"}", str(parallel_default[i] * body_parameter_scale[i])) 

    fp = open(model_path, 'w')
    fp.write(parallel_robot_xml)
    fp.close()

    return model_path

def generate_test_model(body_parameter_scale: np.array, save_path: str,  generations: float):
    os.makedirs(save_path, exist_ok=True)
    test_model_path = os.path.join(save_path, str(generations + 1)+"test.xml")
    mesh_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xml", "stl")).replace("\\", "/")
    parallel_default = [1, 1, 1, 1, 1,
                        0.09742, 0.188, 0.220, # 位置
                        0.052, 0.01, 0.007,  # 质量
                        0.01871796, 0.03525102, 0.03871796, 0.00061102, -0.03988726, -0.00141528, # site的位置
                        -0.01988814, 0.03322523, 0.02116929, -0.03383574, -0.01882983, -0.03383625,
                        ]
    num_param = len(parallel_default)
    parallel_robot_xml ='''
    <mujoco model="parallelrobot">
        <option solver="CG" tolerance="1e-7" timestep="0.001" integrator="implicit"/>
        <compiler meshdir="{MESH_DIR}"/>

        <asset>
            <mesh file="dong1.STL" scale="{1} {1} {3}" />
            <mesh file="dong2.STL" scale="{1} {1} {3}" />
            <mesh file="hinge.STL" />
            <mesh file="jing.STL" scale="{2} {2} {4}" />
            <mesh file="ten.STL" />
            <mesh file="up.STL" scale="{5} {5} 1" />
            <mesh file="down.STL" />
            <texture name="groundplane" type="2d"  builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" mark="edge" markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            <material name="hinge" rgba="0.412 0.412 0.412 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="top"   rgba="0.704 0.502 0.247 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="down"  rgba="0.447 0.455 0.447 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="dong2" rgba="0.75 0.75 0.75 1" reflectance="0.8" specular="1" shininess="0.5"/>
            <material name="jing"  rgba="0.275 0.51 0.706 1" reflectance="1" specular="1" shininess="0.5"/>
            <material name="dong1" rgba="0.275 0.51 0.706 1" reflectance="0.8" specular="1" shininess="0.5"/>
        </asset>
        
        <default>
            <default class="leg-slide">
                <joint type="slide" range="-0.0 0.005" axis="0 0 1" damping="15" stiffness="0" armature="0.01" />
            </default>
            <default class="hin">
                <joint type="hinge" limited="false" damping="100" stiffness="0" armature="0.01" />
            </default>
        </default>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" pos="0 0 0" size="0 0 0.1" type="plane" material="groundplane"/>

            <body name="top" pos="0 0 0.265">
                <freejoint/>
                <geom name="top_geom" type="mesh" mesh="up" material="top" contype="1" conaffinity="1" margin="-0.025" mass="0.173" solref="0.02 1" solimp="0.8 0.9 0.01"/>
                <site name="con1" type="sphere" pos = "{12} {13} -0.005" size="0.008 0.008 0.008"/>
                <site name="con2" type="sphere" pos = "{14} {15} -0.005" size="0.008 0.008 0.008"/>
                <site name="con3" type="sphere" pos = "{16} {17} -0.005" size="0.008 0.008 0.008"/>
                <site name="con4" type="sphere" pos = "{18} {19} -0.005" size="0.008 0.008 0.008"/>
                <site name="con5" type="sphere" pos = "{20} {21} -0.005" size="0.008 0.008 0.008"/>
                <site name="con6" type="sphere" pos = "{22} {23} -0.005" size="0.008 0.008 0.008"/>

                <site name="circle_center" pos="0 0 -0.005" size="0.001"/>
            </body>

            <body name="down" pos="0 0 0">
                <geom type="mesh" mesh="down" pos="0 0 0.015"  material="down" mass="0.623" />
            </body>

            <replicate count="3" euler="0 0 120">
                <body name="leg0" pos="0.07436 0.06631 0.02"> 
                    <geom type="mesh" mesh="hinge" pos="0 0 0" material="hinge" mass="0.007" contype="0" conaffinity="0" condim="1"/>
                    <geom name="ten0" type="mesh" mesh="ten" pos="0 0 0.01" mass="0.00155" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                    <body name="legdown0" pos="0 0 0.01" euler="8 -14 0">               
                        <joint type="hinge" axis="1 0 0" class="hin"/>
                        <joint type="hinge" axis="0 1 0" class="hin"/>
                        <geom type="mesh" mesh="hinge" pos="0 0 0.01" mass="0.007" euler="180 0 90" material="hinge" contype="0" conaffinity="0" />
                        <geom type="mesh" mesh="jing" pos="0 0 0.02" euler="0 0 0" material="jing" mass="{9}"/>
                        <body name="legup0" pos="0 0 0.01" euler="0 0 0">
                            <joint name="slide0" class="leg-slide"/>
                            <joint type="hinge" axis="0 0 1" limited="false" damping="0.5" armature="0.01"/>
                            <geom type="mesh" mesh="dong1" pos="0 0 {6}" euler="0 0 0" material="dong1" contype="1" conaffinity="1" margin="-0.1" mass="{10}" />
                            <geom type="mesh" mesh="dong2" pos="0 0 {7}" euler="0 0 0" material="dong2" contype="1" conaffinity="1" mass="{11}" />
                            <site name="conc0" type="sphere" pos = "0 0 {8}" size="0.008 0.008 0.008" />
                        </body>
                    </body>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                </body>

                <body name="leg1" pos="0.09436 0.03167 0.02">
                    <geom type="mesh" mesh="hinge" pos="0 0 0" mass="0.007" material="hinge" contype="0" conaffinity="0"/>
                    <geom name="ten1" type="mesh" mesh="ten" pos="0 0 0.01" mass="0.00155" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                    <body name="legdown1" pos="0 0 0.01" euler="8 -14 0">
                        <joint type="hinge" axis="1 0 0" class="hin"/>
                        <joint type="hinge" axis="0 1 0" class="hin"/>
                        <geom type="mesh" mesh="hinge" pos="0 0 0.01" mass="0.007" euler="180 0 90" material="hinge" contype="0" conaffinity="0"/>
                        <geom type="mesh" mesh="jing" pos="0 0 0.02" euler="0 0 0" material="jing" mass="{9}" />
                        <body name="legup1" pos="0 0 0.01" euler="0 0 0">
                            <joint name="slide1" class="leg-slide"/> 
                            <joint type="hinge" axis="0 0 1" limited="false" damping="0.5" armature="0.01"/>
                            <geom type="mesh" mesh="dong1" pos="0 0 {6}" euler="0 0 0" material="dong1" contype="1" conaffinity="1" margin="-0.1" mass="{10}" />
                            <geom type="mesh" mesh="dong2" pos="0 0 {7}" euler="0 0 0" material="dong2" contype="1" conaffinity="1" mass="{11}" />
                            <site name="conc1" type="sphere" pos = "0 0 {8}" size="0.008 0.008 0.008"  />
                        </body>
                    </body>
                </body>
            </replicate>
        </worldbody>

        <actuator>
            <motor name="a0" joint="slide00" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a1" joint="slide10" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a2" joint="slide01" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a3" joint="slide11" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a4" joint="slide02" gear="1" damping="0.5" ctrlrange="-10 10"/>
            <motor name="a5" joint="slide12" gear="1" damping="0.5" ctrlrange="-10 10"/>
        </actuator>

        <equality>
            <connect site1="con1" site2="conc00" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con2" site2="conc10" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con3" site2="conc01" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con4" site2="conc11" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con5" site2="conc02" solref="0.001 1" solimp="0.99 0.99 0.001"/>
            <connect site1="con6" site2="conc12" solref="0.001 1" solimp="0.99 0.99 0.001"/>
        </equality>
        
        <sensor>
            <framepos name="center_pos" objtype="site" objname="circle_center"/>
            <framequat name="center_quat" objtype="site" objname="circle_center"/>
            <framelinvel name="center_linvel" objtype="site" objname="circle_center"/>
            <frameangvel name="center_angvel" objtype="site" objname="circle_center"/>
            <framelinacc name="center_linacc" objtype="site" objname="circle_center"/>
            <frameangacc name="center_angacc" objtype="site" objname="circle_center"/>
        </sensor>

    </mujoco>'''

    parallel_robot_xml = parallel_robot_xml.replace("{MESH_DIR}", mesh_dir)
    for i in range(num_param):
        parallel_robot_xml = parallel_robot_xml.replace(
            "{" + str(i + 1) + "}",
            str(parallel_default[i] * body_parameter_scale[i]),
        )

    fp = open(test_model_path, 'w')
    fp.write(parallel_robot_xml)
    fp.close()

    return test_model_path


def spherical_to_cartesian(r, polar, azi, pos):
    polar = math.radians(polar)
    azi = math.radians(azi)
    x = pos[0] + r * np.sin(polar) * np.cos(azi)
    y = pos[1] + r * np.sin(polar) * np.sin(azi)
    z = pos[2] + r * np.cos(polar)
    return [x, y, z]


def compute_euler(up, down):
    vector = up - down
    norm_vector = vector / np.linalg.norm(vector)
    roll = math.degrees(math.atan2(-norm_vector[1], norm_vector[2]))
    pitch = math.degrees(math.asin(norm_vector[0]))
    yaw = 0.0
    return [round(roll, 2), round(pitch, 2), yaw]


def generate_model_v2(xml_path, kuopu_parameter):
    posture = compute_leg_posture(xml_path, kuopu_parameter)
    return apply_leg_posture(xml_path, posture)


def compute_leg_posture(xml_path, kuopu_parameter):
    """Return the posture that should be shared by the training and dynamics models."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)

    up1 = data.site("con1").xpos
    up2 = data.site("con2").xpos
    down1 = data.body("leg00").xpos
    down2 = data.body("leg10").xpos
    length = (np.linalg.norm(up1 - down1) + np.linalg.norm(up2 - down2)) / 2

    # Project search order: [polar1, azimuth1, polar2, azimuth2].
    downpos1 = spherical_to_cartesian(
        length, kuopu_parameter[0], kuopu_parameter[1], up1
    )
    downpos2 = spherical_to_cartesian(
        length, kuopu_parameter[2], kuopu_parameter[3], up2
    )
    euler1 = compute_euler(up1, downpos1)
    euler2 = compute_euler(up2, downpos2)

    return {
        "leg0_pos": downpos1,
        "leg0_euler": euler1,
        "leg1_pos": downpos2,
        "leg1_euler": euler2,
    }


def apply_leg_posture(xml_path, posture):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for body in root.iter("body"):
        name = body.get("name")
        if name == "leg0":
            p = posture["leg0_pos"]
            body.set("pos", f"{p[0]} {p[1]} {p[2]}")
        elif name == "legdown0":
            e = posture["leg0_euler"]
            body.set("euler", f"{e[0]} {e[1]} {e[2]}")
        elif name == "leg1":
            p = posture["leg1_pos"]
            body.set("pos", f"{p[0]} {p[1]} {p[2]}")
        elif name == "legdown1":
            e = posture["leg1_euler"]
            body.set("euler", f"{e[0]} {e[1]} {e[2]}")
    tree.write(xml_path)
    return xml_path
