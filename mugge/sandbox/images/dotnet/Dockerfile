# Toolchain for C#/.NET tickets: dotnet build and xUnit tests. Sandboxes have no network, so the
# packages a project may reference are restored into NUGET_PACKAGES here, at build time:
# projects must use the package versions the xunit template below pins (dotnet list package).
FROM mcr.microsoft.com/dotnet/sdk:9.0
ENV NUGET_PACKAGES=/opt/nuget/packages DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 \
    DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 DOTNET_GENERATE_ASPNET_CERTIFICATE=false
RUN apt-get update && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/* \
    && dotnet new xunit -o /tmp/warm && dotnet new webapi -o /tmp/warm-api \
    && dotnet build /tmp/warm && dotnet build /tmp/warm-api \
    && rm -rf /tmp/warm /tmp/warm-api && chmod -R a+rX /opt/nuget
RUN useradd -m -u 1000 coder 2>/dev/null || true
USER 1000
WORKDIR /work
