```mermaid
graph TD;
    __start__([<p>start</p>]):::first
    node_1(node_1)
    node_2(node_2)
    __end__([<p>end</p>]):::last
    __start__ --> node_1;
    node_1 --> node_2;
    node_2 --> __end__;
    classDef default fill:#ff00,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fcmermaidmermain

