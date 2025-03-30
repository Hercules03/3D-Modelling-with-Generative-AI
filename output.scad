// Camera Lens Model
// Author: OpenSCAD Expert
// Description: A parametric DSLR camera lens with realistic details

// Main parameters
lens_diameter = 70;      // Outer diameter of the lens barrel
lens_length = 100;       // Total length of the lens
mount_diameter = 65;     // Diameter of the lens mount
front_element_diameter = 58; // Diameter of the front glass element
focus_ring_width = 25;   // Width of the focus ring
focus_ring_depth = 3;    // Depth of the knurling on the focus ring
aperture_ring_width = 15; // Width of the aperture ring
aperture_ring_depth = 2; // Depth of the knurling on the aperture ring
knurl_count = 60;        // Number of knurls around the rings

// Colors
color_barrel = [0.2, 0.2, 0.2];    // Black lens barrel
color_ring = [0.25, 0.25, 0.25];   // Dark gray rings
color_glass = [0.8, 0.9, 0.95, 0.6]; // Slightly blue transparent glass
color_mount = [0.3, 0.3, 0.3];     // Dark gray mount
color_red = [0.8, 0.2, 0.2];       // Red accent color for lens markings

// Module for creating the lens barrel
module lens_barrel() {
    color(color_barrel)
    difference() {
        cylinder(h=lens_length, d=lens_diameter, $fn=100);
        translate([0, 0, -1])
            cylinder(h=lens_length+2, d=lens_diameter-4, $fn=100);
    }
}

// Module for lens glass elements
module lens_element(diameter, curvature, thickness) {
    color(color_glass)
    translate([0, 0, 0])
    union() {
        // Main cylindrical part of the element
        cylinder(h=thickness, d=diameter, $fn=100);
        
        // Convex front part
        translate([0, 0, thickness])
            scale([1, 1, curvature])
            sphere(d=diameter, $fn=100);
    }
}

// Module for creating knurled rings (focus and aperture)
module knurled_ring(diameter, width, depth, knurls) {
    color(color_ring)
    difference() {
        cylinder(h=width, d=diameter, $fn=100);
        
        // Create the knurling pattern
        for(i = [0 : knurls - 1]) {
            rotate([0, 0, i * (360 / knurls)])
            translate([diameter/2 - depth/2, 0, -1])
            rotate([0, 0, 45])
            cube([depth, depth, width+2]);
        }
        
        // Hollow out the inside
        translate([0, 0, -1])
            cylinder(h=width+2, d=diameter-8, $fn=100);
    }
}

// Module for lens mount (bayonet style)
module lens_mount() {
    color(color_mount)
    difference() {
        cylinder(h=10, d=mount_diameter, $fn=100);
        
        // Hollow out the inside
        translate([0, 0, -1])
            cylinder(h=12, d=mount_diameter-8, $fn=100);
        
        // Create bayonet notches
        for(i = [0 : 2]) {
            rotate([0, 0, i * 120])
            translate([mount_diameter/2 - 4, 0, 5])
            cube([10, 8, 5], center=true);
        }
    }
    
    // Add mounting pins
    for(i = [0 : 2]) {
        rotate([0, 0, i * 120 + 60])
        translate([mount_diameter/2 - 5, 0, 3])
        rotate([90, 0, 0])
        cylinder(h=3, d=4, center=true, $fn=20);
    }
}

// Module for front lens cap
module lens_cap() {
    color([0.15, 0.15, 0.15])
    difference() {
        union() {
            cylinder(h=3, d=lens_diameter+2, $fn=100);
            cylinder(h=8, d=front_element_diameter+6, $fn=100);
        }
        
        translate([0, 0, -1])
            cylinder(h=10, d=front_element_diameter+2, $fn=100);
    }
}

// Module for branding details
module branding() {
    // Red dot (brand logo)
    color(color_red)
    translate([lens_diameter/2 - 1.5, 0, lens_length - 15])
    rotate([0, 90, 0])
    cylinder(h=3, d=5, $fn=20);
    
    // Lens markings (simplified)
    color([0.9, 0.9, 0.9])
    for(i = [0 : 5]) {
        rotate([0, 0, i * 60])
        translate([lens_diameter/2 - 0.5, 0, lens_length - focus_ring_width - 10])
        cube([1, 1, 3]);
    }
}

// Module for aperture label
module aperture_label() {
    color([0.9, 0.9, 0.9])
    for(i = [0 : 4]) {
        angle = i * 20;
        rotate([0, 0, angle])
        translate([lens_diameter/2 - 0.5, 0, 50])
        cube([1, 1.5, 2]);
        
        // f-numbers
        rotate([0, 0, angle + 10])
        translate([lens_diameter/2 - 0.5, 0, 50])
        cube([1, 0.5, 2]);
    }
}

// Assemble the complete lens
module camera_lens() {
    // Main components
    lens_barrel();
    
    // Front glass element
    translate([0, 0, lens_length - 5])
        lens_element(front_element_diameter, 0.2, 3);
    
    // Secondary glass element
    translate([0, 0, lens_length - 20])
        lens_element(front_element_diameter - 10, -0.15, 2);
    
    // Focus ring
    translate([0, 0, lens_length - focus_ring_width - 20])
        knurled_ring(lens_diameter, focus_ring_width, focus_ring_depth, knurl_count);
    
    // Aperture ring
    translate([0, 0, lens_length - focus_ring_width - aperture_ring_width - 25])
        knurled_ring(lens_diameter - 5, aperture_ring_width, aperture_ring_depth, knurl_count);
    
    // Lens mount at the back
    translate([0, 0, 0])
        lens_mount();
    
    // Branding and labels
    branding();
    aperture_label();
}

// Render the lens
camera_lens();

// Uncomment to show the lens cap
// translate([0, 0, lens_length])
//     lens_cap();