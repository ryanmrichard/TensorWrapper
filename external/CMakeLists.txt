foreach(x ${BACKENDS})
    check_then_build(${x})
endforeach()
