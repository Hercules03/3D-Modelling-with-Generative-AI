// Rubbish Bin Model
// Units are in millimeters

// Parameters for customization
bin_diameter = 250;    // Diameter of the main bin body
bin_height = 350;      // Height of the main bin
wall_thickness = 3;    // Thickness of the bin walls
lid_height = 30;       // Height of the lid
lid_overhang = 5;      // How much the lid extends beyond the bin
fillet_radius = 5;     // Radius for edge fillets
$fn = 100;             // Resolution for curved surfaces

// Main module for the complete bin
module rubbish_bin() {
    color("LightGrey") bin_body();
    color("DarkGrey") translate([0, 0, bin_height]) lid();
}

// Module for the main bin body
module bin_body() {
    difference() {
        // Outer shell
        union() {
            cylinder(d=bin_diameter, h=bin_height);
            
            // Bottom fillet
            translate([0, 0, fillet_radius])
            rotate_extrude()
            translate([bin_diameter/2 - fillet_radius, 0, 0])
            circle(r=fillet_radius);
        }
        
        // Inner hollow
        translate([0, 0, wall_thickness])
        cylinder(d=bin_diameter - 2*wall_thickness, h=bin_height);
        
        // Flat bottom with slight inset
        translate([0, 0, -0.1])
        cylinder(d=bin_diameter - 2*wall_thickness, h=wall_thickness + 0.2);
    }
    
    // Add a base rim for stability
    difference() {
        cylinder(d=bin_diameter, h=wall_thickness);
        translate([0, 0, -0.1])
        cylinder(d=bin_diameter - 20, h=wall_thickness + 0.2);
    }
}

// Module for the bin lid
module lid() {
    difference() {
        union() {
            // Main lid body
            cylinder(d=bin_diameter + 2*lid_overhang, h=lid_height);
            
            // Top dome
            translate([0, 0, lid_height - 0.1])
            scale([1, 1, 0.2])
            sphere(d=bin_diameter + 2*lid_overhang);
            
            // Top fillet
            translate([0, 0, lid_height - fillet_radius])
            rotate_extrude()
            translate([(bin_diameter + 2*lid_overhang)/2 - fillet_radius, 0, 0])
            circle(r=fillet_radius);
        }
        
        // Hollow out the inside
        translate([0, 0, wall_thickness])
        cylinder(d=bin_diameter + 2*lid_overhang - 2*wall_thickness, h=lid_height + 50);
        
        // Create lip to fit onto the bin
        translate([0, 0, -0.1])
        cylinder(d=bin_diameter - 1, h=wall_thickness*2 + 0.1);
    }
}

// Render the complete bin
rubbish_bin();