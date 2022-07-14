#include <vector>
#include <climits>
#include <fast/fast.h>

// This is mechanically generated code.
namespace fast
{

static inline bool test_gt_set(int a, int b, int& min_diff)
{
        if(a > b)
        {
                if(a-b < min_diff)
                        min_diff = a-b;

                return 1;
        }
        return 0;
}

inline int fast_corner_score_10(const fast_byte* cache_0, const int offset[], int b)
{
        b++;
        //This function computes the score for a pixel which is known to be
        //a corner at barrier b. So we start looking at b+1 and above to
        //establish where it stops becoming a corner.
        for(;;)
        {
                int cb = *cache_0 + b;
                int c_b= *cache_0 - b;
                int min_diff = INT_MAX;
                if(test_gt_set(*(cache_0 + offset[0]), cb, min_diff))
                 if(test_gt_set(*(cache_0 + offset[8]), cb, min_diff))
                  if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                   if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                          b += min_diff;
                         else
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           b += min_diff;
                          else
                           break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                               if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                                b += min_diff;
                               else
                                break;
                              else
                               break;
                            else
                             if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                               b += min_diff;
                              else
                               if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                                b += min_diff;
                               else
                                break;
                             else
                              if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                            else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                      if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                     if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                    if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                         else
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                  else
                   if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                      if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                    else
                     break;
                   else
                    break;
                 else if(test_gt_set(c_b, *(cache_0 + offset[8]), min_diff))
                  if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                   if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                         else
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                  else if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                    if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                     if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                              if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                      if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            b += min_diff;
                           else
                            break;
                         else
                          break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else
                      if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                    else
                     if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else
                    if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else
                   if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                 else
                  if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                   if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                     if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else
                    break;
                  else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                  else
                   if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                else if(test_gt_set(c_b, *(cache_0 + offset[0]), min_diff))
                 if(test_gt_set(*(cache_0 + offset[8]), cb, min_diff))
                  if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                   if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                  else if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           break;
                         else
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                    if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           b += min_diff;
                          else
                           break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else
                    if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else
                   if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          break;
                        else
                         if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                 else if(test_gt_set(c_b, *(cache_0 + offset[8]), min_diff))
                  if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                   if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                  else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                    if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             break;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                     if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                           else
                            if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                      if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             b += min_diff;
                            else
                             if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                               if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                                b += min_diff;
                               else
                                break;
                              else
                               break;
                             else
                              break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                          b += min_diff;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                           b += min_diff;
                          else
                           break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                               if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                                b += min_diff;
                               else
                                break;
                              else
                               break;
                             else
                              break;
                            else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             b += min_diff;
                            else
                             if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                               if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                                b += min_diff;
                               else
                                break;
                              else
                               break;
                             else
                              break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else
                      if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                           else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                           else
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                              if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                               b += min_diff;
                              else
                               break;
                             else
                              break;
                            else
                             break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                    else
                     if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                          if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else
                    if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                             if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else
                   if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            b += min_diff;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    break;
                 else
                  if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                      if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                    if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                      if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                            b += min_diff;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                     else
                      break;
                    else
                     if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else
                    if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else
                   break;
                else
                 if(test_gt_set(*(cache_0 + offset[8]), cb, min_diff))
                  if(test_gt_set(*(cache_0 + offset[10]), cb, min_diff))
                   if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                    if(test_gt_set(*(cache_0 + offset[2]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                           b += min_diff;
                          else if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                          if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                     if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            b += min_diff;
                           else
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                         else
                          break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                            b += min_diff;
                           else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                           else
                            if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                             if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                              b += min_diff;
                             else
                              break;
                            else
                             break;
                          else
                           break;
                         else
                          break;
                        else if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         if(test_gt_set(*(cache_0 + offset[3]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                    if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else
                    if(test_gt_set(*(cache_0 + offset[14]), cb, min_diff))
                     if(test_gt_set(*(cache_0 + offset[6]), cb, min_diff))
                      if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                       if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                        if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                        if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(*(cache_0 + offset[15]), cb, min_diff))
                         if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                          if(test_gt_set(*(cache_0 + offset[9]), cb, min_diff))
                           if(test_gt_set(*(cache_0 + offset[13]), cb, min_diff))
                            if(test_gt_set(*(cache_0 + offset[7]), cb, min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else
                   break;
                 else if(test_gt_set(c_b, *(cache_0 + offset[8]), min_diff))
                  if(test_gt_set(c_b, *(cache_0 + offset[10]), min_diff))
                   if(test_gt_set(*(cache_0 + offset[4]), cb, min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                   else if(test_gt_set(c_b, *(cache_0 + offset[4]), min_diff))
                    if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                     if(test_gt_set(*(cache_0 + offset[12]), cb, min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[1]), cb, min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[11]), cb, min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                         if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                           b += min_diff;
                          else
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            b += min_diff;
                           else
                            break;
                         else
                          if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                        else
                         break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      if(test_gt_set(c_b, *(cache_0 + offset[2]), min_diff))
                       if(test_gt_set(c_b, *(cache_0 + offset[1]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[3]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                    else
                     break;
                   else
                    if(test_gt_set(c_b, *(cache_0 + offset[14]), min_diff))
                     if(test_gt_set(c_b, *(cache_0 + offset[6]), min_diff))
                      if(test_gt_set(c_b, *(cache_0 + offset[12]), min_diff))
                       if(test_gt_set(*(cache_0 + offset[5]), cb, min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else if(test_gt_set(c_b, *(cache_0 + offset[5]), min_diff))
                        if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            b += min_diff;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                       else
                        if(test_gt_set(c_b, *(cache_0 + offset[15]), min_diff))
                         if(test_gt_set(c_b, *(cache_0 + offset[13]), min_diff))
                          if(test_gt_set(c_b, *(cache_0 + offset[7]), min_diff))
                           if(test_gt_set(c_b, *(cache_0 + offset[9]), min_diff))
                            if(test_gt_set(c_b, *(cache_0 + offset[11]), min_diff))
                             b += min_diff;
                            else
                             break;
                           else
                            break;
                          else
                           break;
                         else
                          break;
                        else
                         break;
                      else
                       break;
                     else
                      break;
                    else
                     break;
                  else
                   break;
                 else
                  break;

        }

        return b-1;
}

void fast_corner_score_10(
  const fast_byte* img,
  const int img_stride,
  const std::vector<fast_xy>& corners,
  const int threshold,
  std::vector<int>& scores)
{
  scores.resize(corners.size());
  int pixel[16] = {
    0 + img_stride * 3,
    1 + img_stride * 3,
    2 + img_stride * 2,
    3 + img_stride * 1,
    3 + img_stride * 0,
    3 + img_stride * -1,
    2 + img_stride * -2,
    1 + img_stride * -3,
    0 + img_stride * -3,
    -1 + img_stride * -3,
    -2 + img_stride * -2,
    -3 + img_stride * -1,
    -3 + img_stride * 0,
    -3 + img_stride * 1,
    -2 + img_stride * 2,
    -1 + img_stride * 3,
  };
  for(unsigned int n=0; n < corners.size(); n++)
    scores[n] = fast_corner_score_10(img + corners[n].y*img_stride + corners[n].x, pixel, threshold);
}

} // namespace Fast
