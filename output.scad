// Hexagonal Pencil Holder
// All measurements in millimeters

// Main parameters
height = 100;        // Height of the pencil holder
outer_radius = 60;   // Radius to the vertices of the hexagon
wall_thickness = 3;  // Wall thickness
bottom_thickness = 5; // Bottom thickness
sides = 6;           // Number of sides for the hexagon

// Derived parameters
inner_radius = outer_radius - wall_thickness;

// Create the hexagonal pencil holder
difference() {
    // Outer hexagonal prism
    cylinder(h=height, r=outer_radius, $fn=sides);
    
    // Inner hexagonal cutout
    translate([0, 0, bottom_thickness])
    cylinder(h=height, r=inner_radius, $fn=sides);
}

// Optional: Add a small bevel at the top edge
difference() {
    // Main body already created above
    
    // Cut a small chamfer at the top edge
    translate([0, 0, height - 2])
    difference() {
        cylinder(h=3, r=outer_radius + 5, $fn=sides);
        translate([0, 0, -0.5])
        cylinder(h=4, r1=outer_radius - 1, r2=outer_radius - 3, $fn=sides);
    }
}