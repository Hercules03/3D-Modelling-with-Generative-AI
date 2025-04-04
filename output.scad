/* 
 * SHAPE-SHIFTING DRONE WITH SLIDER-BASED GRASPING MECHANISM
 * 
 * This OpenSCAD model implements a drone design with:
 * - A modular frame that can transform between configurations
 * - An integrated slider-based grasping mechanism
 * - Aerodynamic considerations for flight efficiency
 */

// MAIN PARAMETERS
// Frame dimensions
frame_width = 200;      // Width of main frame
frame_height = 30;      // Height of main frame
frame_thickness = 4;    // Thickness of frame walls

// Arm parameters
arm_length = 120;       // Length of each arm
arm_width = 15;         // Width of each arm
arm_height = 10;        // Height of each arm
arm_count = 4;          // Number of arms

// Motor mount parameters
motor_mount_diameter = 25;  // Diameter of motor mount
motor_mount_height = 8;     // Height of motor mount

// Battery compartment
battery_length = 80;
battery_width = 40;
battery_height = 20;

// Grasping mechanism
gripper_length = 120;    // Length of gripper mechanism
gripper_width = 80;      // Width of gripper assembly
slider_length = 80;      // Length of slider rails
slider_width = 8;        // Width of slider rails
slider_height = 6;       // Height of slider rails
gripper_finger_length = 30; // Length of gripper fingers
gripper_finger_width = 8;   // Width of gripper fingers

// Transformation mechanism
joint_diameter = 15;
joint_height = 12;
pivot_diameter = 6;

// Aesthetic parameters
rounding_radius = 3;     // Radius for rounded corners
$fn = 50;                // Resolution for circular objects

// MAIN ASSEMBLY
module drone() {
    // Central body
    union() {
        central_frame();
        
        // Attach arms in X configuration
        for (i = [0:3]) {
            rotate([0, 0, i * 90])
            translate([frame_width/2 - 10, 0, 0])
            arm();
        }
        
        // Add battery compartment
        translate([0, 0, -frame_height/2 - battery_height/2])
        battery_compartment();
        
        // Add grasping mechanism
        translate([0, 0, -frame_height/2 + 5])
        grasping_mechanism();
    }
}

// CENTRAL FRAME
module central_frame() {
    difference() {
        // Main body
        minkowski() {
            cube([frame_width - 2*rounding_radius, 
                  frame_width - 2*rounding_radius, 
                  frame_height - 2*rounding_radius], center = true);
            sphere(r = rounding_radius);
        }
        
        // Hollow out the inside
        minkowski() {
            cube([frame_width - 2*frame_thickness - 2*rounding_radius, 
                  frame_width - 2*frame_thickness - 2*rounding_radius, 
                  frame_height + 1 - 2*rounding_radius], center = true);
            sphere(r = rounding_radius);
        }
        
        // Cut out space for transformation mechanism
        for (i = [0:3]) {
            rotate([0, 0, i * 90])
            translate([frame_width/2 - 15, 0, 0])
            cylinder(h = frame_height + 2, d = joint_diameter, center = true);
        }
    }
    
    // Add transformation joints
    for (i = [0:3]) {
        rotate([0, 0, i * 90])
        translate([frame_width/2 - 15, 0, 0])
        transformation_joint();
    }
    
    // Add shape-shifting pivot points
    for (i = [0:3]) {
        rotate([0, 0, i * 90 + 45])
        translate([frame_width/2 - 25, 0, 0])
        shape_shifting_pivot();
    }
}

// ARM MODULE
module arm() {
    difference() {
        union() {
            // Arm body
            translate([arm_length/2, 0, 0])
            minkowski() {
                cube([arm_length - 2*rounding_radius, 
                      arm_width - 2*rounding_radius, 
                      arm_height - 2*rounding_radius], center = true);
                sphere(r = rounding_radius);
            }
            
            // Joint connector
            cylinder(h = arm_height, d = joint_diameter + 5, center = true);
            
            // Add motor mount at the end
            translate([arm_length, 0, arm_height/2])
            motor_mount();
        }
        
        // Cutout for joint pivot
        cylinder(h = arm_height + 2, d = pivot_diameter, center = true);
        
        // Weight reduction cutouts
        for (i = [0:2]) {
            translate([30 + i * 30, 0, 0])
            cylinder(h = arm_height + 1, d = 8, center = true);
        }
    }
}

// MOTOR MOUNT
module motor_mount() {
    difference() {
        // Base
        cylinder(h = motor_mount_height, d = motor_mount_diameter, center = true);
        
        // Center hole for motor shaft
        cylinder(h = motor_mount_height + 1, d = 8, center = true);
        
        // Mounting holes
        for (i = [0:3]) {
            rotate([0, 0, i * 90])
            translate([motor_mount_diameter/2 - 4, 0, 0])
            cylinder(h = motor_mount_height + 1, d = 3, center = true);
        }
    }
}

// BATTERY COMPARTMENT
module battery_compartment() {
    difference() {
        // Main compartment
        minkowski() {
            cube([battery_length - 2*rounding_radius, 
                  battery_width - 2*rounding_radius, 
                  battery_height - 2*rounding_radius], center = true);
            sphere(r = rounding_radius);
        }
        
        // Hollow inside
        translate([0, 0, 2])
        minkowski() {
            cube([battery_length - 2*frame_thickness - 2*rounding_radius, 
                  battery_width - 2*frame_thickness - 2*rounding_radius, 
                  battery_height - 2 - 2*rounding_radius], center = true);
            sphere(r = rounding_radius);
        }
        
        // Access door
        translate([0, 0, -battery_height/2 + 0.5])
        cube([battery_length - 10, battery_width - 10, 2], center = true);
    }
    
    // Mounting points to main frame
    for (x = [-1, 1]) {
        for (y = [-1, 1]) {
            translate([x * (battery_length/2 - 8), y * (battery_width/2 - 8), battery_height/2])
            cylinder(h = 5, d = 6, center = true);
        }
    }
}

// GRASPING MECHANISM
module grasping_mechanism() {
    translate([0, -gripper_length/4, 0]) {
        difference() {
            union() {
                // Base plate
                minkowski() {
                    cube([gripper_width - 2*rounding_radius, 
                          gripper_length - 2*rounding_radius, 
                          5 - 2*rounding_radius], center = true);
                    sphere(r = rounding_radius);
                }
                
                // Slider rails
                for (x = [-1, 1]) {
                    translate([x * (gripper_width/2 - slider_width/2), 0, 2.5])
                    slider_rail();
                }
                
                // Fixed gripper end
                translate([0, -gripper_length/2 + gripper_finger_length/2, 10])
                fixed_gripper_end();
            }
            
            // Cutout for moving parts
            translate([0, gripper_length/4, 0])
            cube([gripper_width - 20, gripper_length/2, 10], center = true);
        }
        
        // Add the moving slider
        translate([0, gripper_length/4, 5])
        sliding_gripper();
    }
}

// SLIDER RAIL
module slider_rail() {
    minkowski() {
        cube([slider_width - 2*rounding_radius, 
              slider_length - 2*rounding_radius, 
              slider_height - 2*rounding_radius], center = true);
        sphere(r = rounding_radius);
    }
}

// FIXED GRIPPER END
module fixed_gripper_end() {
    difference() {
        union() {
            // Base
            cube([gripper_width - 20, gripper_finger_width, 10], center = true);
            
            // Gripper fingers
            for (x = [-1, 1]) {
                translate([x * (gripper_width/2 - 20), gripper_finger_length/2, 0])
                rotate([0, 0, x * 15])
                cube([8, gripper_finger_length, 10], center = true);
            }
        }
        
        // Finger texture for better grip
        for (x = [-1, 1]) {
            for (i = [0:3]) {
                translate([x * (gripper_width/2 - 20), 5 + i * 5, 0])
                rotate([0, 0, x * 15])
                cube([9, 2, 6], center = true);
            }
        }
    }
}

// SLIDING GRIPPER
module sliding_gripper() {
    difference() {
        union() {
            // Base for sliding mechanism
            minkowski() {
                cube([gripper_width - 25 - 2*rounding_radius, 
                      30 - 2*rounding_radius, 
                      10 - 2*rounding_radius], center = true);
                sphere(r = rounding_radius);
            }
            
            // Gripper fingers
            for (x = [-1, 1]) {
                translate([x * (gripper_width/2 - 20), -gripper_finger_length/2, 0])
                rotate([0, 0, -x * 15])
                cube([8, gripper_finger_length, 10], center = true);
            }
            
            // Slider connectors
            for (x = [-1, 1]) {
                translate([x * (gripper_width/2 - slider_width/2), 0, -5])
                cube([slider_width, 20, 3], center = true);
            }
        }
        
        // Slider channel cutouts
        for (x = [-1, 1]) {
            translate([x * (gripper_width/2 - slider_width/2), 0, -5])
            cube([slider_width - 2, 25, 4], center = true);
        }
        
        // Finger texture for better grip
        for (x = [-1, 1]) {
            for (i = [0:3]) {
                translate([x * (gripper_width/2 - 20), -5 - i * 5, 0])
                rotate([0, 0, -x * 15])
                cube([9, 2, 6], center = true);
            }
        }
    }
}

// TRANSFORMATION JOINT
module transformation_joint() {
    difference() {
        union() {
            cylinder(h = joint_height, d = joint_diameter, center = true);
            
            // Reinforcement
            for (i = [0:1]) {
                rotate([0, 0, i * 90])
                cube([joint_diameter, 5, joint_height], center = true);
            }
        }
        
        // Center pivot hole
        cylinder(h = joint_height + 1, d = pivot_diameter, center = true);
    }
}

// SHAPE-SHIFTING PIVOT
module shape_shifting_pivot() {
    difference() {
        // Pivot base
        cylinder(h = 8, d = 12, center = true);
        
        // Pivot hole
        cylinder(h = 9, d = 4, center = true);
    }
    
    // Locking mechanisms for different configurations
    for (i = [0:3]) {
        rotate([0, 0, i * 90])
        translate([8, 0, 0])
        cylinder(h = 6, d = 3, center = true);
    }
}

// Render the drone
drone();