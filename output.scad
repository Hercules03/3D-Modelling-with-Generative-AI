// Basic Drum Model in OpenSCAD
// Features a cylindrical shell, two drumheads, tension rods, and lugs

// Parameters for customization
shell_diameter = 300;  // Drum shell diameter in mm
shell_height = 200;    // Drum shell height in mm
shell_thickness = 6;   // Shell wall thickness in mm
head_thickness = 2;    // Drumhead thickness in mm
rim_height = 15;       // Height of the rim in mm
rim_thickness = 3;     // Thickness of the rim in mm
lug_count = 8;         // Number of tension lugs around the drum
tension_rod_diameter = 5; // Diameter of tension rods

// Main drum shell module
module drum_shell() {
    difference() {
        cylinder(h=shell_height, d=shell_diameter, center=true);
        cylinder(h=shell_height+1, d=shell_diameter-2*shell_thickness, center=true);
    }
}

// Drumhead module (slightly curved to simulate tension)
module drumhead(is_top) {
    translate([0, 0, is_top ? shell_height/2 : -shell_height/2]) {
        difference() {
            union() {
                // Slightly curved drumhead
                translate([0, 0, is_top ? 0 : -head_thickness])
                    scale([1, 1, 0.1]) 
                        sphere(d=shell_diameter);
                
                // Flat edge for connection to rim
                cylinder(h=head_thickness, d=shell_diameter, center=true);
            }
            // Cut off the bottom/top part to create just a dome
            translate([0, 0, is_top ? -shell_diameter/2 : shell_diameter/2])
                cube([shell_diameter*2, shell_diameter*2, shell_diameter], center=true);
        }
    }
}

// Rim module
module rim(is_top) {
    translate([0, 0, is_top ? shell_height/2 : -shell_height/2]) {
        difference() {
            cylinder(h=rim_height, d=shell_diameter+2*rim_thickness, center=true);
            cylinder(h=rim_height+1, d=shell_diameter, center=true);
        }
    }
}

// Tension lug module
module tension_lug(is_top) {
    height_pos = is_top ? shell_height/2 : -shell_height/2;
    
    translate([0, 0, height_pos]) {
        rotate([0, is_top ? 0 : 180, 0]) {
            translate([shell_diameter/2 + rim_thickness, 0, 0]) {
                cube([20, 15, 30], center=true);
                // Lug hook
                translate([5, 0, -10])
                    rotate([0, 90, 0])
                        cylinder(h=10, d=8, center=true);
            }
        }
    }
}

// Tension rod module
module tension_rod(angle, is_top) {
    height_pos = is_top ? shell_height/2 + rim_height/2 : -shell_height/2 - rim_height/2;
    
    rotate([0, 0, angle]) {
        translate([shell_diameter/2 + rim_thickness/2, 0, height_pos]) {
            rotate([0, 90, 0])
                cylinder(h=25, d=tension_rod_diameter, center=true);
            
            // Rod end (tension screw head)
            translate([12, 0, 0])
                rotate([0, 90, 0])
                    cylinder(h=4, d=10, center=true);
        }
    }
}

// Assemble the complete drum
module complete_drum() {
    // Shell
    color("BurlyWood") drum_shell();
    
    // Top and bottom drumheads
    color("White", 0.8) {
        drumhead(true);  // Top head
        drumhead(false); // Bottom head
    }
    
    // Rims
    color("Silver") {
        rim(true);  // Top rim
        rim(false); // Bottom rim
    }
    
    // Tension lugs and rods
    color("DarkGray") {
        for(angle = [0:360/lug_count:359]) {
            rotate([0, 0, angle]) {
                tension_lug(true);
                tension_lug(false);
            }
        }
    }
    
    // Tension rods
    color("Silver") {
        for(angle = [0:360/lug_count:359]) {
            tension_rod(angle, true);
            tension_rod(angle, false);
        }
    }
}

// Create the drum
complete_drum();