#version 430 core
#extension GL_ARB_explicit_attrib_location : require


//in
layout(location = 0) in vec3 pos_in;
// layout(location = 5) in float log_depth_in;


//out
layout(location = 0) out vec3 pos_out;

// //uniform

void main(){

    pos_out=pos_in;
  
}

