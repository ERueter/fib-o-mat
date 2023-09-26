//
// Created by yorgo on 07/06/20.
//

#ifndef SVGIMPORT_GROUP_CONTEXT_H
#define SVGIMPORT_GROUP_CONTEXT_H

#include "base_context.h"

namespace svgimport
{
    class group_context : public base_context
    {
    public:
        group_context(base_context &parent)
            : base_context(parent)
        {
            // std::cout << "group" << "\n";
        }

        void on_exit_element()
        {
            // do nothing for now
        }

//        template <typename T>
//        void set(svgpp::tag::attribute::id, T id)
//        {
//            std::cout << id << "\n";
//        }
    };
}

#endif //SVGIMPORT_GROUP_CONTEXT_H
