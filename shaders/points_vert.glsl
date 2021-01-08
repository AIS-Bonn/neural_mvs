#version 430 core
#extension GL_ARB_explicit_attrib_location : require


//in
in vec3 position;

//out
layout(location = 0) out vec3 pos_out;

//uniforms
uniform mat4 MVP;


void main(){

   

    gl_Position = MVP*vec4(position, 1.0);

    pos_out=position;

}

