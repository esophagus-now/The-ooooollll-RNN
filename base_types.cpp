#include <iostream>
#include "base_types.h"

std::ostream& operator<<(std::ostream &o, layer const& l) {
    l.dump(o);
    return o;
}

