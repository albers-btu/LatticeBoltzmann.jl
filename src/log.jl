using Dates
using Logging

# Tee stderr + file via stdlib SimpleLogger (no custom AbstractLogger: that
# hits a world-age error after the first kernel compile).
struct TeeIO <: IO
    console::IO
    file::IO
end
function Base.unsafe_write(t::TeeIO, p::Ptr{UInt8}, n::UInt)
    nout = unsafe_write(t.console, p, n)
    unsafe_write(t.file, p, n)
    flush(t.file)
    return nout
end
function Base.write(t::TeeIO, x::UInt8)
    write(t.console, x)
    n = write(t.file, x)
    flush(t.file)
    return n
end
function Base.write(t::TeeIO, xs::StridedVector{UInt8})
    write(t.console, xs)
    n = write(t.file, xs)
    flush(t.file)
    return n
end
Base.flush(t::TeeIO) = (flush(t.console); flush(t.file); nothing)
Base.isopen(t::TeeIO) = isopen(t.console) && isopen(t.file)
Base.close(t::TeeIO) = close(t.file)

const _RUN_LOG = Ref{Any}(nothing)

function _logstate_key()
    isdefined(Base.CoreLogging, :LOGGER_STATE) && return Base.CoreLogging.LOGGER_STATE
    return :LOGGER_STATE
end

function _install_logger(logger::AbstractLogger)
    global_logger(logger)
    # REPL / include() uses a task-local logger; global_logger alone is ignored.
    try
        task_local_storage(_logstate_key(), Base.CoreLogging.LogState(logger))
    catch
    end
    return logger
end

function start_run_log!(dir::AbstractString; filename::AbstractString="run.log")
    if _RUN_LOG[] !== nothing
        return _RUN_LOG[].path
    end
    mkpath(dir)
    path = joinpath(dir, filename)
    io = open(path, "a")
    println(io, "===== ", Dates.now(), " pid=", getpid(), " =====")
    flush(io)
    prev = current_logger()
    logger = SimpleLogger(TeeIO(stderr, io), Logging.Info)
    _install_logger(logger)
    _RUN_LOG[] = (io=io, prev=prev, path=path)
    @info "logging to $path"
    return path
end

function stop_run_log!()
    state = _RUN_LOG[]
    state === nothing && return nothing
    _RUN_LOG[] = nothing
    global_logger(state.prev)
    try
        delete!(task_local_storage(), _logstate_key())
    catch
    end
    println(state.io, "===== end ", Dates.now(), " =====")
    close(state.io)
    return nothing
end

atexit(stop_run_log!)
