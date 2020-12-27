#ifndef DEBUG_H
#define DEBUG_H 1

#include <iostream>
#include <fstream>

#define DEBUG(tag, thing)                              \
    debug_out << "{\n\t\"tag\":\"" << (tag) << "\",\n" \
              << "\t\"file\":\"" __FILE__ "\",\n"      \
              << "\t\"line\":" << __LINE__ << ",\n"    \
              << "\t\"data\":" << (thing) << "\n},\n"

extern std::ofstream debug_out;

template <typename T>
struct dump {
    T const& the_object;

    dump(T const& the_object) : the_object(the_object) {}
};

template <typename T>
std::ostream& operator<<(std::ostream& o, dump<T> const& d) {
    d.the_object.dump(o);
    return o;
}

#endif