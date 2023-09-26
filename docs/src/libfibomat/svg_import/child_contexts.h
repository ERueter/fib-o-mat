//
// Created by yorgo on 07/06/20.
//

#ifndef SVGIMPORT_CHILD_CONTEXTS_H
#define SVGIMPORT_CHILD_CONTEXTS_H

#include <svgpp/svgpp.hpp>

#include "base_context.h"
#include "basic_shape_context.h"
#include "group_context.h"
#include "path_context.h"
#include "traits.h"

namespace svgimport
{
    struct child_context_factories
    {
        template <class ParentContext, class ElementTag, class Enable = void>
        struct apply
        {
            typedef svgpp::factory::context::on_stack<base_context> type;
        };
    };

    template <>
    struct child_context_factories::apply<base_context, svgpp::tag::element::g, void>
    {
        typedef svgpp::factory::context::on_stack<group_context> type;
    };

    typedef boost::mpl::set<base_context, group_context> svg_g_parent_contexts;

    template <class ParentContext, class ElementTag>
    struct child_context_factories::apply<
        ParentContext, ElementTag,
        typename boost::enable_if_c<
            boost::mpl::has_key<basic_shape_elements, ElementTag>::value
            && boost::mpl::has_key<svg_g_parent_contexts, ParentContext>::value
        >::type
    >
    {
        typedef svgpp::factory::context::on_stack<basic_shape_context> type;
    };

    template <class ParentContext, class ElementTag>
    struct child_context_factories::apply<
        ParentContext, ElementTag,
        typename boost::enable_if_c<
            boost::mpl::has_key<path_elements, ElementTag>::value
            && boost::mpl::has_key<svg_g_parent_contexts, ParentContext>::value
        >::type
    >
    {
        typedef svgpp::factory::context::on_stack<path_context> type;
    };

//    template <class ElementTag>
//    struct child_context_factories::apply<group_context, ElementTag, typename boost::enable_if<boost::mpl::has_key<svgimport::basic_shape_elements, ElementTag>>::type>
//    {
//        typedef svgpp::factory::context::on_stack<basic_shape_context> type;
//    };
}

#endif //SVGIMPORT_CHILD_CONTEXTS_H
