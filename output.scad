// Parametric Tabletop Base with Magnet Holes
// Created: April 2025

// Main parameters
base_diameter = 40;    // Diameter of the base in mm
base_height = 3;       // Height/thickness of the base in mm
lip_width = 4;         // Width of the lip around the base in mm
lip_height = 1;        // Height of the lip in mm

// Magnet hole parameters
magnet_diameter = 2.5; // Diameter of the magnet holes in mm
magnet_depth = 1.5;    // Depth of the magnet holes in mm
magnet_offset = 12;    // Distance from center to magnet centers in mm

// Resolution parameter
$fn = 100;             // Smoothness of curved surfaces

// Main module for the tabletop base
module tabletop_base() {
    difference() {
        // Main body with lip
        union() {
            // Base cylinder
            cylinder(h=base_height, d=base_diameter);
            
            // Lip ring
            difference() {
                // Outer cylinder for the lip
                cylinder(h=base_height+lip_height, d=base_diameter);
                
                // Inner cutout to create the lip
                translate([0, 0, base_height])
                    cylinder(h=lip_height+0.1, d=base_diameter-2*lip_width);
            }
        }
        
        // Create three magnet holes in a triangular pattern
        for (i = [0:120:359]) {
            rotate([0, 0, i])
                translate([magnet_offset, 0, base_height-magnet_depth])
                    cylinder(h=magnet_depth+0.1, d=magnet_diameter, $fn=50);
        }
    }
}

// Render the model
tabletop_base();