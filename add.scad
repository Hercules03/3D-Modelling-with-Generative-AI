module motorHolders(x, diameter, height) {
  motorHolder(x, x, diameter, height); 
  motorHolder(-x, x, diameter, height); 
  motorHolder(x, -x, diameter, height); 
  motorHolder(-x, -x, diameter, height); 
}

module motorHoles(x, diameter, height) {
  motorHole(x, x, diameter, height); 
  motorHole(x, -x, diameter, height); 
  motorHole(-x, x, diameter, height); 
  motorHole(-x, -x, diameter, height); 
}

module motorHolder(x, y, motorDiameter, height = 3) {
  motorRadius = motorDiameter / 2;
  translate([x, y, 0])
  linear_extrude(height)
  circle(motorRadius + 2);
}

module motorHole(x, y, motorDiameter, height = 3) {
  motorRadius = motorDiameter / 2;
  translate([x, y, -1])
  linear_extrude(height + 2)
  circle(motorRadius);
}


module sides(x, sideOffset, h) {
  sideTB(x, sideOffset, 1, h);
  sideTB(x, sideOffset, -1, h);
  sideLR(x, sideOffset, 1, h);
  sideLR(x, sideOffset, -1, h);
}

module sideLR(x, sideOffset, dir, h) {
  p0=[dir * x, x - sideOffset];
  p1=[dir * sideOffset, 0];
  p2=[dir * x, -x + sideOffset];
  linear_extrude(h)
  BezConic( p0, p1, p2, steps=20);
}

module sideTB(x, sideOffset, dir, h) {
  p0=[x - sideOffset, dir * x];
  p1=[0, dir * sideOffset];
  p2=[-x + sideOffset, dir * x];
  linear_extrude(h)
  BezConic( p0, p1, p2, steps=20);
}

module cross(x, sideOffset, dir, h) {
  p0 = [dir * x, x - sideOffset];
  p1 = [dir * (x - sideOffset), x];
  p2 = [dir * -x, -x + sideOffset];
  p3 = [dir * (-x + sideOffset), -x];

  linear_extrude(h)
  polygon([p0, p1, p2, p3]);
}

module centerHole(h) {
  union() {
    translate([-10, -6, -1])
    cube([20, 12, h + 2]);
    translate([-8, -(23.5 / 2) + 1, -1])
    cube([16, 22.5, h + 2]);

    translate([-8, -(23.5 / 2) - 8, -1])
    cube([16, 8, h + 2]);
  }
}

module strapHoles(h) {
  union() {

    translate([-22, 10, -1])
    cylinder(r = 2, h = h + 3);  

    translate([22, 10, -1])
    cylinder(r = 2, h = h + 3);  
  
    translate([-22, -10, -1])
    cylinder(r = 2, h = h + 3);  

    translate([22, -10, -1])
    cylinder(r = 2, h = h + 3);  

    translate([-24, 9.5, -1])
    cylinder(r = 1, h = h + 3);  

    translate([24, 9.5, -1])
    cylinder(r = 1, h = h + 3);  
  
    translate([-24, -9.5, -1])
    cylinder(r = 1, h = h + 3);  

    translate([24, -9.5, -1])
    cylinder(r = 1, h = h + 3);  


  }
}

module BezConic(p0,p1,p2,steps=5) {

	stepsize1 = (p1-p0)/steps;
	stepsize2 = (p2-p1)/steps;

	for (i=[0:steps-1]) {
		assign(point1 = p0+stepsize1*i) 
		assign(point2 = p1+stepsize2*i) 
		assign(point3 = p0+stepsize1*(i+1))
		assign(point4 = p1+stepsize2*(i+1))  {
			assign( bpoint1 = point1+(point2-point1)*(i/steps) )
			assign( bpoint2 = point3+(point4-point3)*((i+1)/steps) ) {
				polygon(points=[bpoint1,bpoint2,p1]);
			}
		}
	}
}

$fn = 30;

motorDistance = 120; // motor to motor distance
motorDiameter = 8.8; // hole size for 8.5mm motor
sideOffset = 3;

motorHolderHeight = 4;
crossHeight = 2;
sideHeight = 2;

x = sqrt(pow(motorDistance / 2, 2) * 2) / 2;

difference() {

  mainBody();

  motorHoles(x, motorDiameter, motorHolderHeight);
  centerHole(crossHeight);

  strapHoles(crossHeight);
}

module mainBody() {

  // motors
  motorHolders(x, motorDiameter, motorHolderHeight); 

  // cross beams
  cross(x, sideOffset, 1, crossHeight);
  cross(x, sideOffset, -1, crossHeight);

  // curved sides
  sides(x, sideOffset, sideHeight);
  
  boardMount(sideHeight);
}

module boardMount(h) {

  union() {
    translate([-12, (23.5 / 2), 0])
    cube([4, 2, h + 2]);

    translate([8, (23.5 / 2), 0])
    cube([4, 2, h + 2]);

    translate([-12, -(23.5 / 2), 0])
    cube([2, 23.5, h + 1]);

    translate([10, -(23.5 / 2), 0])
    cube([2, 23.5, h + 1]);


    translate([-10, - (23.5 / 2) - 10, 0])
    cube([4, 2, h + 2]);

    translate([6, - (23.5 / 2) - 10, 0])
    cube([4, 2, h + 2]);
  }
}