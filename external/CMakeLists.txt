foreach(x ${BACKENDS_ON})
    check_then_build(${x})
endforeach()
