/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 7, 2015
 */
/**
 * Based on code by
 *     Daniel Lemire, https://github.com/lemire/FastPFor
 * which was available under the Apache License, Version 2.0.
 */

#ifndef CODECS_CODECFACTORY_H_
#define CODECS_CODECFACTORY_H_ 

#include "Rice.h"
#include "OptRice.h"

#include "varintGB.h"
#include "varintG8IU.h"
#include "varintG8CU.h"

#include "Simple9.h"
#include "Simple16.h"

#include "NewPFor.h"
#include "OptPFor.h"


namespace Codecs {

using CodecMap = std::map<std::string, std::shared_ptr<IntegerCodec> >;

/**
 * This class is a convenience class to generate codecs quickly.
 * It cannot be used safely in a multithreaded context where
 * each thread should have a different IntegerCodec.
 */
class CodecFactory {
public:
    static CodecMap scodecmap;

    // hacked for convenience
    static std::vector<std::shared_ptr<IntegerCodec> > allSchemes() {
        std::vector<std::shared_ptr<IntegerCodec> > ans;
        for (auto i = scodecmap.begin(); i != scodecmap.end(); ++i) {
            ans.push_back(i->second);
        }
        return ans;
    }

    static std::vector<std::string> allNames() {
        std::vector<std::string> ans;
        for (auto i = scodecmap.begin(); i != scodecmap.end(); ++i) {
            ans.push_back(i->first);
        }
        return ans;
    }

    static std::shared_ptr<IntegerCodec> getFromName(const std::string &name) {
        if (scodecmap.find(name) == scodecmap.end()) {
            std::cerr << "name " << name << " does not refer to a CODEC." << std::endl;
            std::cerr << "possible choices:" << std::endl;
            for (auto i = scodecmap.begin(); i != scodecmap.end(); ++i) {
                std::cerr << static_cast<std::string>(i->first) << std::endl; // useless cast, but just to be clear
            }
            std::cerr << "for now, I'm going to just return 'copy'" << std::endl;
            return std::shared_ptr<IntegerCodec>(new JustCopy);
        }
        return scodecmap[name];
    }
};

} // namespace Codecs

#endif // CODECS_CODECFACTORY_H_
