#ifndef __LIST_H__
#define __LIST_H__

#include "commondef.h"


typedef struct list {
  struct list* next;
} list_t;

/* List Node
 * ptr     a pointer to a struct instance which embeds a list node.
 * type    the type of the struct instance.
 * name    the name of the embedded list_t member
 *
 * Trivial means of extracting a node.
 * It's true that we can just use
 * ptr->self instead of ptr + offsetof(type, self).
 * The rationale for using the latter is just
 * to keep consistency with its partner "list_base".
 * Plus, this works with both void* and type*.
 */

#define list_node(ptr, type, name)												\
  (list_t*)((uint8_t*)(ptr) + offsetof(type, name))

/* List Base
 * node_ptr    a pointer to a list element which is embedded within a struct instance.
 * type        the struct type which the node is embedded in.
 * name        the name of the embedded list_t member
 *
 * We get a pointer to the base of the memory layout "type", from "node_ptr".
 */
#define list_base(node_ptr, type, name)											\
  (type*)((uint8_t*)(node_ptr) - offsetof(type, name))

/* List Push Function Implementation
 * pproot    double pointer to root node
 * pn        pointer to node which is to be pushed.
 * type      the type which contains the node
 * name      the name of the embedded list_t member  
 *
 * This is a subroutine which is designed to be embedded
 * in an arbitrary function with a parameter list of (type** pproot, type* pn)
 */
#define list_push_fn_impl(pproot, pn, type, name)				\
  do {                                                  \
    if (*(pproot) == NULL) {														\
      *(pproot) = (pn);                                 \
    } else {                                            \
      list_t* rlist = list_node(*(pproot), type, name);	\
      list_t* end = list_end(rlist);                    \
      end->next = &(pn)->self;                          \
    }                                                   \
  } while (0)

/* List Free Function Node
 *
 * Parameter list is essentially the same as List Free Function Implementation.
 * This is only used in the macro defined below, and exists
 * for the purposes of readability.
 */
#define list_free_fn_node(pnode, type, free_func, name) \
  do {                                             \
    if ((pnode) != NULL) {													 \
      type* ptype = list_base(pnode, type, name);        \
      (free_func)(ptype);                          \
      free(ptype);                                 \
      (pnode) = NULL;                              \
    }                                              \
  } while (0)

/* List Free Function Implementation
 * proot     pointer to the root of the list
 * type      the type which contains the node
 * free_func the name of a function which takes
 *           a pointer to type "type" and frees its
 *           (internal) memory.
 *
 * Designed to be embedded in a function of parameter list of (type* root).
 * Frees the internal memory for each member within the list.
 */

#define list_free_fn_impl(proot, type, free_func, name)                       \
  do {                                                                  \
    if ((proot) == NULL) {                                                     \
      return;                                                           \
    }                                                                   \
    list_t* n = NULL;                                                   \
    list_t* start = list_node(proot, type, name);                             \
    for (list_t* piter = start; piter != NULL; piter = piter->next) {   \
      list_free_fn_node(n, type, free_func, name);											\
      n = piter;                                                        \
    }                                                                   \
    list_free_fn_node(n, type, free_func, name);                              \
  } while (0)

/* List End
 * n an arbitrary list node.
 * ret the first node which has a NULL next member.
 */

static inline list_t* list_end(list_t* n) {
  return (n != NULL && n->next != NULL) ? list_end(n->next) : n;
}

#endif // __LIST_H__
