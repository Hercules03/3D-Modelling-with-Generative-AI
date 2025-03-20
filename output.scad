openscad
// Paper Box OpenSCAD Code
// Dimensions:
// Outer shell: 250mm x 200mm x 300mm
// Wall thickness: 5mm

// Create hollow rectangular prism (outer shell)
$outer = rotated([45, 1]) square(250) * 300;
$inner = cylinder(h=270, r=220);

// Subtract inner part to create hollow walls
$paperBox = union() {
    $outer - $inner;
};

// Add handle on front face (top middle)
$handle = rotate([90, 180]) cylinder(h=305, r=4);
translate([0, 270, 125]) $handle;

// Scale for better visualization
scale(2) $paperBox;