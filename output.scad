// DSLR Camera Model
// Author: OpenSCAD Expert
// Description: A parametric DSLR camera with lens, body, and controls

/* Main Parameters */
camera_body_width = 130;
camera_body_height = 90;
camera_body_depth = 60;
camera_roundness = 5;
lens_diameter = 60;
lens_length = 80;
grip_depth = 25;

/* Camera Body Module */
module camera_body() {
    difference() {
        // Main body
        hull() {
            translate([camera_roundness, camera_roundness, camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_body_width-camera_roundness, camera_roundness, camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_roundness, camera_body_height-camera_roundness, camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_body_width-camera_roundness, camera_body_height-camera_roundness, camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_roundness, camera_roundness, camera_body_depth-camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_body_width-camera_roundness, camera_roundness, camera_body_depth-camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_roundness, camera_body_height-camera_roundness, camera_body_depth-camera_roundness])
                sphere(r=camera_roundness);
            translate([camera_body_width-camera_roundness, camera_body_height-camera_roundness, camera_body_depth-camera_roundness])
                sphere(r=camera_roundness);
        }
        
        // Grip cutout
        translate([camera_body_width-grip_depth/2, camera_body_height/2, camera_body_depth/2])
            rotate([0, 90, 0])
                cylinder(h=grip_depth, r=camera_body_height/2.2, center=true, $fn=60);
    }
    
    // Add camera grip
    translate([camera_body_width-grip_depth*0.8, camera_body_height/2, camera_body_depth/2])
        rotate([0, 90, 0])
            difference() {
                cylinder(h=grip_depth*0.6, r=camera_body_height/2, center=true, $fn=60);
                cylinder(h=grip_depth*0.7, r=camera_body_height/2.5, center=true, $fn=60);
                translate([0, 0, -grip_depth])
                    cube([camera_body_height*2, camera_body_height*2, grip_depth*2], center=true);
            }
}

/* Lens Module */
module lens() {
    // Lens barrel
    difference() {
        union() {
            // Main lens cylinder
            cylinder(h=lens_length, d=lens_diameter, $fn=60);
            
            // Lens rings
            for (i = [0:3]) {
                translate([0, 0, 10 + i * 20])
                    cylinder(h=3, d=lens_diameter + 5, $fn=60);
            }
            
            // Lens mount
            translate([0, 0, -5])
                cylinder(h=5, d1=lens_diameter + 15, d2=lens_diameter, $fn=60);
        }
        
        // Front lens element
        translate([0, 0, lens_length-2])
            cylinder(h=5, d=lens_diameter-10, $fn=60);
    }
}

/* Viewfinder Module */
module viewfinder() {
    // Viewfinder base
    translate([camera_body_width/2, camera_body_height, camera_body_depth/4])
        difference() {
            union() {
                // Prism housing
                cube([30, 20, 15], center=true);
                
                // Eyepiece
                translate([0, 15, 0])
                    rotate([90, 0, 0])
                        cylinder(h=10, d=15, $fn=40);
            }
            
            // Eyepiece hole
            translate([0, 15, 0])
                rotate([90, 0, 0])
                    cylinder(h=12, d=10, $fn=40);
        }
}

/* Camera Controls Module */
module controls() {
    // Mode dial
    translate([15, camera_body_height, 15])
        difference() {
            cylinder(h=5, d=20, $fn=40);
            for (i = [0:7]) {
                rotate([0, 0, i * 45])
                    translate([8, 0, 0])
                        cylinder(h=6, d=2, $fn=20);
            }
        }
    
    // Shutter button
    translate([camera_body_width-30, camera_body_height, 15])
        union() {
            cylinder(h=2, d=15, $fn=30);
            translate([0, 0, 2])
                cylinder(h=3, d=10, $fn=30);
        }
        
    // Control wheel
    translate([camera_body_width-15, camera_body_height-15, camera_body_depth-5])
        difference() {
            cylinder(h=3, d=20, $fn=40);
            for (i = [0:11]) {
                rotate([0, 0, i * 30])
                    translate([8, 0, 0])
                        cylinder(h=4, d=2, $fn=20);
            }
        }
}

/* LCD Screen Module */
module lcd_screen() {
    // LCD frame
    translate([camera_body_width/2, camera_body_height/2, camera_body_depth])
        difference() {
            cube([camera_body_width-20, camera_body_height-20, 2], center=true);
            cube([camera_body_width-40, camera_body_height-40, 3], center=true);
        }
}

/* Flash Module */
module flash() {
    // Flash housing
    translate([camera_body_width/2, camera_body_height, camera_body_depth/2])
        union() {
            cube([30, 5, 20], center=true);
            translate([0, 5, 0])
                cube([20, 10, 15], center=true);
        }
}

/* Assemble the Camera */
module camera() {
    camera_body();
    
    // Position the lens
    translate([camera_body_width/2, camera_body_height/2, 0])
        rotate([0, 90, 0])
            lens();
    
    viewfinder();
    controls();
    lcd_screen();
    flash();
}

// Render the camera
camera();